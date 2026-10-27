import pandas as pd
import pytest

from voc_rag import preparacao


@pytest.fixture
def tabelas():
    """Mini Olist: 5 reviews, uma só com nota, uma só pontuação, uma repetida em 2 pedidos."""
    reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4", "r4"],
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "review_score": [1, 5, 3, 2, 2],
        "review_comment_title": [None, "Ótimo", None, "Não chegou", "Não chegou"],
        "review_comment_message": ["Veio quebrado\r\nliguem (11) 98877-1234", "chegou rápido",
                                   None, "...", "..."],
        "review_creation_date": ["2018-03-01"] * 5,
        "review_answer_timestamp": ["2018-03-02 10:00:00"] * 5,
    })
    pedidos = pd.DataFrame({
        "order_id": ["o1", "o2", "o3", "o4", "o5"],
        "customer_id": ["c1", "c2", "c3", "c4", "c5"],
        "order_status": ["delivered"] * 4 + ["shipped"],
        "order_purchase_timestamp": ["2018-02-01 09:00:00"] * 5,
        "order_delivered_customer_date": ["2018-02-20 12:00:00", "2018-02-05 12:00:00",
                                          None, "2018-02-10 12:00:00", None],
        "order_estimated_delivery_date": ["2018-02-15 00:00:00"] * 5,
    })
    clientes = pd.DataFrame({"customer_id": ["c1", "c2", "c3", "c4", "c5"],
                             "customer_state": ["SP", "RJ", "MG", "SP", "SP"],
                             "customer_city": ["sao paulo"] * 5})
    itens = pd.DataFrame({"order_id": ["o1", "o1", "o1", "o2", "o4"],
                          "order_item_id": [1, 2, 3, 1, 1],
                          "product_id": ["p1", "p2", "p2", "p1", "p3"],
                          "price": [10.0] * 5, "freight_value": [1.0] * 5})
    produtos = pd.DataFrame({"product_id": ["p1", "p2", "p3"],
                             "product_category_name": ["beleza_saude", "pet_shop", None]})
    traducao = pd.DataFrame({"product_category_name": ["beleza_saude", "pet_shop"],
                             "product_category_name_english": ["health_beauty", "pet_shop"]})
    return {"reviews": reviews, "pedidos": pedidos, "clientes": clientes, "itens": itens,
            "produtos": produtos, "traducao": traducao}


def test_montar_texto():
    assert preparacao.montar_texto("Recomendo", "Chegou rápido") == "Recomendo. Chegou rápido"
    assert preparacao.montar_texto("Ótimo!", "Chegou") == "Ótimo! Chegou"
    assert preparacao.montar_texto(float("nan"), "  ") == ""


def test_selecionar_com_texto_descarta_so_nota(tabelas):
    com_texto, n = preparacao.selecionar_com_texto(tabelas["reviews"])
    assert n == {"total": 5, "so_nota": 1, "sem_letra": 0, "com_texto": 4}
    assert "r3" not in set(com_texto["review_id"])


def test_deduplicar_junta_pedidos(tabelas):
    com_texto, _ = preparacao.selecionar_com_texto(tabelas["reviews"])
    unicas = preparacao.deduplicar_reviews(com_texto)
    assert unicas["review_id"].is_unique
    assert unicas.set_index("review_id").loc["r4", "order_ids"] == "o4|o5"


def test_categoria_mais_frequente_e_sem_categoria(tabelas):
    cat = preparacao.categoria_por_pedido(tabelas["itens"], tabelas["produtos"],
                                          tabelas["traducao"]).set_index("order_id")
    assert cat.loc["o1", "categoria"] == "pet_shop"      # 2 de 3 itens
    assert cat.loc["o1", "n_categorias"] == 2
    assert cat.loc["o4", "categoria"] == "sem_categoria"


def test_construir_base_ponta_a_ponta(tabelas, tmp_path):
    docs = preparacao.construir_base(tabelas).set_index("review_id")
    assert list(docs.index) == ["r1", "r2", "r4"]
    assert docs.loc["r1", "texto"] == "Veio quebrado liguem [TELEFONE]"
    assert docs.loc["r1", "pii_mascarada"] == "TELEFONE"
    assert docs.loc["r1", "sentimento_nota"] == "negativo"
    assert docs.loc["r1", "atraso_dias"] == 5 and docs.loc["r1", "entregue_com_atraso"]
    assert not docs.loc["r2", "entregue_com_atraso"]
    assert docs.loc["r4", "texto"] == "Não chegou. ..."
    assert docs.loc["r4", "texto_lexical"] == "nao chegou. ..."
    assert docs.loc["r4", "categoria"] == "sem_categoria"

    caminho = preparacao.salvar_base(docs.reset_index(), tmp_path / "base.csv")
    relida = preparacao.carregar_base(caminho)
    assert len(relida) == 3
    assert pd.api.types.is_datetime64_any_dtype(relida["data_review"])


def test_validar_base_reclama_de_review_id_duplicado(tabelas):
    docs = preparacao.construir_base(tabelas)
    with pytest.raises(ValueError, match="review_id duplicado"):
        preparacao.validar_base(pd.concat([docs, docs.head(1)]))


def test_salvar_base_usa_fim_de_linha_unix(tmp_path):
    """Mesmo arquivo (e mesmo hash no manifesto) no Windows e no Linux."""
    caminho = preparacao.salvar_base(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), tmp_path / "b.csv")
    assert caminho.read_bytes() == b"a,b\n1,x\n2,y\n"
