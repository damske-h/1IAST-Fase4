"""Gold set da avaliação do RAG: perguntas, resultado esperado e partição dev/teste.

Sete estratos, um por comportamento que o RAG precisa ter:

| Estrato | O que testa | Esperado |
|---|---|---|
| A agregada | tema com centenas de reviews relevantes | respondida; precisão@10 e cobertura de subtemas |
| B agulha | um fato que aparece em poucas reviews | respondida; a review-alvo no top-10 |
| C paráfrase | a mesma agulha com outras palavras | respondida; a review-alvo no top-10 |
| D metadado | filtro na própria pergunta (UF, período, nota, categoria) | respondida; filtros extraídos certos |
| E sem evidência | do domínio, mas sem resposta na base | recusada |
| F fora de escopo | nada a ver com reviews | recusada |
| G adversarial | injeção, PII, dado de terceiro | bloqueada (ou recusada) |

A partição separa um teste **trancado**: só roda uma vez, no fim, para o relatório.
"""
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from voc_rag.busca import tokenizar_lexical
from voc_rag.config import RAIZ, SEMENTE
from voc_rag.schemas import FiltrosBusca

ARQUIVO_GOLD = RAIZ / "avaliacao" / "gold_set.csv"
ESTRATOS = {"A": "agregada", "B": "agulha", "C": "paráfrase", "D": "metadado",
            "E": "sem evidência", "F": "fora de escopo", "G": "adversarial"}


class ItemGold(BaseModel):
    """Uma pergunta do gold set, validada."""

    id: str = Field(pattern=r"^[A-G]\d{2}$")
    estrato: Literal["A", "B", "C", "D", "E", "F", "G"]
    particao: Literal["dev", "teste"]
    pergunta: str = Field(min_length=3)
    esperado: Literal["respondida", "recusada", "bloqueada"]
    filtros: FiltrosBusca = Field(default_factory=FiltrosBusca)   # D: os filtros que o roteador deve extrair
    subtemas: list[str] = Field(default_factory=list)             # A: o que uma boa resposta cobre
    review_alvo: str | None = None                                # B e C: a review que precisa ser achada
    observacao: str = ""

    @field_validator("filtros", mode="before")
    @classmethod
    def ler_json(cls, valor):
        return json.loads(valor) if isinstance(valor, str) and valor.strip() else (valor or {})

    @field_validator("subtemas", mode="before")
    @classmethod
    def ler_lista(cls, valor):
        if isinstance(valor, str):
            return [s.strip() for s in valor.split("|") if s.strip()]
        return valor or []


def carregar_gold(caminho=ARQUIVO_GOLD, particao=None):
    """Lê e valida o gold set. `particao` = "dev", "teste" ou None (todas)."""
    tabela = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    itens = [ItemGold(**{k: (v or None) if k == "review_alvo" else v for k, v in linha.items()})
             for linha in tabela.to_dict("records")]
    ids = [i.id for i in itens]
    if len(ids) != len(set(ids)):
        raise ValueError("id repetido no gold set")
    faltando_alvo = [i.id for i in itens if i.estrato in "BC" and not i.review_alvo]
    if faltando_alvo:
        raise ValueError(f"perguntas B/C sem review_alvo: {faltando_alvo}")
    return [i for i in itens if particao is None or i.particao == particao]


def particionar(estratos, frac_teste=1 / 3, semente=SEMENTE):
    """Sorteia a partição dentro de cada estrato (estratificada e reprodutível)."""
    rng = np.random.default_rng(semente)
    estratos = pd.Series(list(estratos))
    particao = pd.Series("dev", index=estratos.index)
    for _, grupo in estratos.groupby(estratos):
        n_teste = round(len(grupo) * frac_teste)
        particao[rng.choice(grupo.index, size=n_teste, replace=False)] = "teste"
    return particao.tolist()


def candidatos_agulha(documentos, n=40, min_palavras=12, max_df=15, semente=SEMENTE):
    """Reviews boas para perguntas-agulha: longas e com termos raros na base.

    Um termo que aparece em até `max_df` reviews é específico ("garantia estendida",
    "voltagem"): uma pergunta sobre ele tem poucas respostas certas, e dá para
    conferir se a busca acha a review-alvo. Sorteia equilibrando as notas.
    """
    tokens = documentos["texto_lexical"].map(lambda t: set(tokenizar_lexical(t)))
    df = pd.Series([tok for conj in tokens for tok in conj]).value_counts()
    raros = set(df[(df <= max_df) & (df.index.str.len() >= 5)].index)
    candidatos = documentos.assign(
        termos_raros=tokens.map(lambda conj: sorted(conj & raros)),
        n_palavras=documentos["texto"].str.split().str.len(),
    )
    candidatos = candidatos[(candidatos["n_palavras"] >= min_palavras)
                            & (candidatos["termos_raros"].str.len() >= 2)
                            & ~candidatos["texto"].str.contains(r"\[(?:EMAIL|CPF|TELEFONE|CEP)\]")]
    por_nota = max(1, n // 5)
    amostra = (candidatos.sample(frac=1, random_state=semente)       # embaralha e pega
               .groupby("nota").head(por_nota).sort_values("nota"))  # até por_nota de cada nota
    return amostra[["review_id", "nota", "categoria", "estado", "ano_mes", "termos_raros", "texto"]] \
        .reset_index(drop=True)


def resumo(itens):
    """Contagem por estrato e partição, para conferir o desenho."""
    tabela = pd.DataFrame([{"estrato": i.estrato, "particao": i.particao} for i in itens])
    return pd.crosstab(tabela["estrato"].map(lambda e: f"{e} {ESTRATOS[e]}"), tabela["particao"],
                       margins=True, margins_name="total")


def contem_quantificacao(texto):
    """Quantificação sem base: "a maioria", "muitos", "70%"... A métrica própria do protocolo.

    O RAG vê 10 reviews, não a base: afirmar proporção é inventar número.
    """
    return bool(re.search(r"\b(maioria|maior parte|muitos|muitas|quase todos|grande parte|"
                          r"poucos|poucas|frequentemente|geralmente)\b|\d+\s?%",
                          texto, re.IGNORECASE))

