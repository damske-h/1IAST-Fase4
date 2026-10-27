"""Preparação da base de conhecimento: reviews da Olist -> documentos com metadados.

Cada função é uma etapa do notebook 01, onde as decisões são explicadas.
Para gerar a base sem abrir o notebook:
    python -m voc_rag.preparacao
"""
import re
from pathlib import Path

import pandas as pd

from voc_rag.config import ARQUIVO_BASE
from voc_rag.texto import limpar_leve, limpar_normalizar, mascarar_pii

# Colunas da base final: nome na Olist -> nome na base de conhecimento
COLUNAS = {
    # identificação e citação
    "review_id": "review_id", "order_ids": "order_ids",
    # texto
    "texto": "texto", "texto_lexical": "texto_lexical",
    "review_comment_title": "titulo_original", "review_comment_message": "mensagem_original",
    # filtros
    "review_score": "nota", "sentimento_nota": "sentimento_nota",
    "review_creation_date": "data_review", "review_answer_timestamp": "data_resposta",
    "ano_mes": "ano_mes", "order_status": "status_pedido",
    "order_purchase_timestamp": "data_compra", "order_delivered_customer_date": "data_entrega",
    "order_estimated_delivery_date": "data_entrega_estimada",
    "dias_entrega": "dias_entrega", "atraso_dias": "atraso_dias",
    "entregue_com_atraso": "entregue_com_atraso",
    "categoria": "categoria", "categoria_en": "categoria_en", "n_categorias": "n_categorias",
    "n_itens": "n_itens", "valor_produtos": "valor_produtos", "valor_frete": "valor_frete",
    "customer_state": "estado", "customer_city": "cidade",
    "n_palavras": "n_palavras", "pii_mascarada": "pii_mascarada",
}
COLUNAS_DATA = ["data_review", "data_resposta", "data_compra", "data_entrega",
                "data_entrega_estimada"]


# --- Seção 2: seleção -----------------------------------------------------------
def montar_texto(titulo, mensagem):
    """Junta título e mensagem num único texto; ausentes viram string vazia."""
    partes = [str(p).strip() for p in (titulo, mensagem) if isinstance(p, str) and p.strip()]
    if len(partes) == 2 and not re.search(r"[.!?…]$", partes[0]):
        partes[0] += "."           # "Recomendo" + "Chegou rápido" -> "Recomendo. Chegou rápido"
    return " ".join(partes)


def selecionar_com_texto(reviews):
    """Fica só com as reviews cujo título + mensagem tem ao menos uma letra.

    Devolve (reviews com a coluna texto_bruto, contagens para o relatório).
    """
    reviews = reviews.copy()
    reviews["texto_bruto"] = [montar_texto(t, m) for t, m in
                              zip(reviews["review_comment_title"], reviews["review_comment_message"])]
    tem_letra = reviews["texto_bruto"].str.contains(r"[A-Za-zÀ-ÿ]", regex=True)
    contagens = {
        "total": len(reviews),
        "so_nota": int((reviews["texto_bruto"] == "").sum()),
        "sem_letra": int(((reviews["texto_bruto"] != "") & ~tem_letra).sum()),
        "com_texto": int(tem_letra.sum()),
    }
    return reviews[tem_letra].copy(), contagens


# --- Seção 3: duplicatas ----------------------------------------------------------
def deduplicar_reviews(reviews):
    """Uma linha por review_id; os pedidos da mesma review ficam em order_ids ("a|b")."""
    pedidos_por_review = (reviews.groupby("review_id")["order_id"]
                          .agg(lambda ids: "|".join(sorted(set(ids)))).rename("order_ids"))
    return (reviews.sort_values(["review_id", "order_id"])
            .drop_duplicates("review_id", keep="first")
            .merge(pedidos_por_review, on="review_id")
            .reset_index(drop=True))


# --- Seção 4: metadados -----------------------------------------------------------
def categoria_por_pedido(itens, produtos, traducao):
    """Categoria mais frequente entre os itens do pedido (empate: a do primeiro item)."""
    itens = itens.sort_values(["order_id", "order_item_id"])
    itens_cat = (itens.merge(produtos[["product_id", "product_category_name"]], on="product_id", how="left")
                 .merge(traducao, on="product_category_name", how="left"))
    itens_cat["product_category_name"] = itens_cat["product_category_name"].fillna("sem_categoria")
    return itens_cat.groupby("order_id", sort=False).agg(
        categoria=("product_category_name", lambda c: c.value_counts().index[0]),
        categoria_en=("product_category_name_english",
                      lambda c: c.dropna().value_counts().index[0] if c.notna().any() else None),
        n_categorias=("product_category_name", "nunique"),
        n_itens=("order_item_id", "count"),
        valor_produtos=("price", "sum"),
        valor_frete=("freight_value", "sum"),
    ).reset_index()


def juntar_metadados(reviews, pedidos, clientes, categorias):
    """reviews -> pedidos -> clientes, e reviews -> categoria do pedido (left joins)."""
    base = (reviews
            .merge(pedidos, on="order_id", how="left")
            .merge(clientes[["customer_id", "customer_state", "customer_city"]],
                   on="customer_id", how="left")
            .merge(categorias, on="order_id", how="left"))
    base["categoria"] = base["categoria"].fillna("sem_categoria")
    if len(base) != len(reviews):
        raise ValueError("a junção duplicou reviews: alguma chave não é única")
    return base


# --- Seção 5: limpeza ---------------------------------------------------------------
def limpar_textos(base):
    """Cria `texto` (leve + PII mascarada), `pii_mascarada` e `texto_lexical` (para BM25)."""
    base = base.copy()
    limpos = [mascarar_pii(limpar_leve(t)) for t in base["texto_bruto"]]
    base["texto"] = [t for t, _ in limpos]
    base["pii_mascarada"] = [",".join(sorted(set(a))) for _, a in limpos]
    base["texto_lexical"] = base["texto"].map(lambda t: limpar_normalizar(t, remove_acentos=True))
    return base


# --- Seção 6: metadados derivados -----------------------------------------------------
def derivar_metadados(base):
    """Sentimento pela nota, prazos de entrega, período e tamanho do texto."""
    base = base.copy()
    for col in ["review_creation_date", "review_answer_timestamp", "order_purchase_timestamp",
                "order_delivered_customer_date", "order_estimated_delivery_date"]:
        base[col] = pd.to_datetime(base[col], errors="coerce")

    base["sentimento_nota"] = pd.cut(base["review_score"], bins=[0, 2, 3, 5],
                                     labels=["negativo", "neutro", "positivo"]).astype(str)
    base["dias_entrega"] = (base["order_delivered_customer_date"]
                            - base["order_purchase_timestamp"]).dt.days
    base["atraso_dias"] = (base["order_delivered_customer_date"].dt.normalize()
                           - base["order_estimated_delivery_date"].dt.normalize()).dt.days
    base["entregue_com_atraso"] = (base["atraso_dias"] > 0).astype("boolean")
    base.loc[base["atraso_dias"].isna(), "entregue_com_atraso"] = pd.NA   # não entregue
    base["ano_mes"] = base["review_creation_date"].dt.strftime("%Y-%m")
    base["n_palavras"] = base["texto"].str.split().str.len()
    return base


# --- Seção 8: validação e gravação ------------------------------------------------------
def selecionar_colunas(base):
    """Fica só com as colunas da base final, já com os nomes em português."""
    return base[list(COLUNAS)].rename(columns=COLUNAS)


def validar_base(documentos):
    """Garantias que o resto do pipeline assume. Levanta ValueError se alguma quebrar."""
    checagens = {
        "review_id duplicado: a citação ficaria ambígua": documentos["review_id"].is_unique,
        "documento sem texto": documentos["texto"].str.len().gt(0).all(),
        "nota fora de 1..5": documentos["nota"].between(1, 5).all(),
        "quebra de linha sobrou no texto":
            not documentos["texto"].str.contains(r"\\r\\n|\r|\n", regex=True).any(),
    }
    falhas = [msg for msg, ok in checagens.items() if not ok]
    if falhas:
        raise ValueError("; ".join(falhas))


def salvar_base(documentos, caminho=ARQUIVO_BASE):
    """Grava a base em CSV (UTF-8, sem índice, fim de linha "\n").

    O fim de linha fixo deixa o arquivo (e o hash no manifesto do índice) igual
    no Windows e no Linux; o padrão do pandas é o do sistema operacional.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    documentos.to_csv(caminho, index=False, encoding="utf-8", lineterminator="\n")
    return caminho


def carregar_base(caminho=ARQUIVO_BASE):
    """Relê a base com as datas como datetime. É assim que o notebook 02 abre o CSV."""
    documentos = pd.read_csv(caminho, parse_dates=COLUNAS_DATA)
    documentos["pii_mascarada"] = documentos["pii_mascarada"].fillna("")  # CSV não distingue "" de ausente
    return documentos


# --- Pipeline completo ----------------------------------------------------------------
def construir_base(tabelas):
    """Roda todas as etapas sobre o dicionário de tabelas de `voc_rag.dados.carregar_tabelas`."""
    com_texto, _ = selecionar_com_texto(tabelas["reviews"])
    com_texto = deduplicar_reviews(com_texto)
    categorias = categoria_por_pedido(tabelas["itens"], tabelas["produtos"], tabelas["traducao"])
    base = juntar_metadados(com_texto, tabelas["pedidos"], tabelas["clientes"], categorias)
    base = derivar_metadados(limpar_textos(base))
    documentos = selecionar_colunas(base)
    validar_base(documentos)
    return documentos


if __name__ == "__main__":
    from voc_rag.dados import baixar_dados, carregar_tabelas

    documentos = construir_base(carregar_tabelas(baixar_dados()))
    caminho = salvar_base(documentos)
    print(f"{len(documentos):,} documentos gravados em {caminho}")
