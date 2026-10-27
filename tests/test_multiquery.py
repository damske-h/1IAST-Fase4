import pandas as pd
import pytest

from tests.llm_falso import LLMFalso
from tests.modelo_falso import ModeloFalso
from voc_rag import busca, indice, multiquery
from voc_rag.texto import limpar_normalizar


@pytest.fixture
def buscador():
    textos = ["Paguei o boleto e o pedido não foi liberado", "Chegou rápido, recomendo muito",
              "Veio quebrado e mal embalado", "O pagamento no boleto demorou a compensar",
              "Ainda não recebi o pedido"]
    docs = pd.DataFrame({
        "review_id": [f"r{i}" for i in range(5)], "texto": textos,
        "texto_lexical": [limpar_normalizar(t, remove_acentos=True) for t in textos],
        "nota": [1, 5, 1, 2, 2], "categoria": ["a"] * 5, "estado": ["SP"] * 5,
        "ano_mes": ["2018-01"] * 5,
        "entregue_com_atraso": pd.array([pd.NA] * 5, dtype="boolean"),
    })
    vetores = indice.gerar_embeddings(ModeloFalso(), docs["texto"])
    return busca.Buscador(docs, indice.IndiceVetorial(vetores, docs["review_id"]), ModeloFalso())


def test_limpar_variacoes():
    assert multiquery.limpar_variacoes("boleto", ["Boleto", " pagar  boleto ", "", "pagar boleto",
                                                  "a", "b", "c"], qtd=3) == ["pagar boleto", "a", "b"]


def test_gerador_usa_schema_e_delimitador():
    llm = LLMFalso({"consultas": ["boleto não compensou", "boleto", "pagamento por boleto"]})
    gerador = multiquery.criar_gerador_variacoes(llm, qtd=2)
    assert gerador.invoke({"consulta": "boleto"}) == ["boleto não compensou", "pagamento por boleto"]
    assert "<consulta>boleto</consulta>" in llm.prompts[0] and "Gere 2" in llm.prompts[0]


def test_gerar_consultas_sobrevive_a_falha():
    consultas, avisos = multiquery.gerar_consultas(multiquery.criar_gerador_variacoes(LLMFalso({})), "frete")
    assert consultas == ["frete"] and "falhou" in avisos[0]


def test_buscar_multi_query_funde_por_review_id(buscador):
    res = multiquery.buscar_multi_query(buscador, ["boleto", "pagamento boleto pedido"], k=3)
    assert res["review_id"].is_unique and len(res) <= 3
    assert {"r0", "r3"} <= set(res["review_id"])
    assert res["n_consultas"].max() == 2
    assert res["pontuacao"].is_monotonic_decreasing
    assert res.attrs["consultas"] == ["boleto", "pagamento boleto pedido"]


def test_buscar_multi_query_respeita_permitidos(buscador):
    res = multiquery.buscar_multi_query(buscador, ["boleto", "pedido"], k=5, permitidos=[3, 4])
    assert set(res["review_id"]) <= {"r3", "r4"}
