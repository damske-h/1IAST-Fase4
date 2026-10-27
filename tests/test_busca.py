import numpy as np
import pandas as pd
import pytest

from tests.modelo_falso import ModeloFalso
from voc_rag import avaliacao, busca, indice
from voc_rag.texto import limpar_normalizar


@pytest.fixture
def documentos():
    textos = ["Não chegou até hoje", "Chegou rápido, recomendo", "Veio quebrado",
              "Preciso da nota fiscal", "Ainda não recebi o pedido"]
    return pd.DataFrame({
        "review_id": [f"r{i}" for i in range(5)],
        "texto": textos,
        "texto_lexical": [limpar_normalizar(t, remove_acentos=True) for t in textos],
        "nota": [1, 5, 1, 3, 2],
        "categoria": ["a", "b", "a", "b", "a"],
        "estado": ["SP", "RJ", "SP", "MG", "SP"],
        "ano_mes": ["2018-01", "2018-02", "2018-03", "2018-04", "2018-05"],
        "entregue_com_atraso": pd.array([pd.NA, False, True, False, pd.NA], dtype="boolean"),
    })


@pytest.fixture
def buscador(documentos):
    vetores = indice.gerar_embeddings(ModeloFalso(), documentos["texto"])
    return busca.Buscador(documentos, indice.IndiceVetorial(vetores, documentos["review_id"]), ModeloFalso())


def test_rrf_premia_quem_aparece_bem_nos_dois_rankings():
    fundido = busca.rrf([[1, 2, 3], [3, 1, 4]])
    assert [p for p, _ in fundido][:2] == [1, 3]
    assert fundido[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_bm25_normaliza_a_consulta_como_a_base(documentos):
    bm25 = busca.IndiceBM25(documentos["texto_lexical"])
    resultado = bm25.buscar("NOTA FISCAL", k=3)
    assert resultado[0][0] == 3
    assert bm25.buscar("não", k=5)[0][0] in (0, 4)        # "não" casa com "nao"
    assert bm25.buscar("xyzw", k=3) == []                  # nenhum termo: nada volta


def test_bm25_respeita_permitidos(documentos):
    bm25 = busca.IndiceBM25(documentos["texto_lexical"])
    assert [p for p, _ in bm25.buscar("nao", k=5, permitidos=[4])] == [4]


def test_posicoes_permitidas(documentos):
    assert list(busca.posicoes_permitidas(documentos, nota_max=2)) == [0, 2, 4]
    assert list(busca.posicoes_permitidas(documentos, categorias=["b"], nota_min=4)) == [1]
    assert list(busca.posicoes_permitidas(documentos, periodo=("2018-02", "2018-03"))) == [1, 2]
    assert list(busca.posicoes_permitidas(documentos, entregue_com_atraso=True)) == [2]
    assert list(busca.posicoes_permitidas(documentos, entregue_com_atraso=False)) == [1, 3]


@pytest.mark.parametrize("modo", busca.MODOS)
def test_buscador_devolve_top_k_com_citacao(buscador, modo):
    resultado = buscador.buscar("nota fiscal", k=3, modo=modo)
    assert 1 <= len(resultado) <= 3
    assert list(resultado.index[:1]) == [1]
    assert {"review_id", "similaridade", "pos_vetorial", "pos_bm25", "nota"} <= set(resultado.columns)
    assert resultado.attrs["modo"] == modo and resultado.attrs["latencia_ms"] >= 0


def test_buscador_hibrido_com_filtro(buscador, documentos):
    permitidos = busca.posicoes_permitidas(documentos, nota_max=2)
    resultado = buscador.buscar("nao chegou", k=5, permitidos=permitidos)
    assert set(resultado["review_id"]) <= {"r0", "r2", "r4"}


def test_buscador_recusa_modo_invalido(buscador):
    with pytest.raises(ValueError):
        buscador.buscar("x", modo="magico")


def test_checagens(buscador):
    perguntas = [("nota fiscal", r"nota fiscal")]
    tabela = avaliacao.checar_recuperacao(buscador, perguntas, k=1)
    assert set(tabela["modo"]) == set(busca.MODOS)
    assert tabela.set_index("modo").loc["bm25", "precisao@1"] == 1.0

    recusa = avaliacao.checar_recusa(buscador, perguntas, ["zzz"], limiar=0.5)
    assert list(recusa["grupo"]) == ["dentro", "fora"]
    assert np.issubdtype(recusa["similaridade_top1"].dtype, np.floating)


def test_min_palavras_tira_reviews_curtas_da_busca(documentos):
    documentos.loc[5] = ["r5", "Ótimo", "otimo", 5, "b", "RJ", "2018-06", pd.NA]
    vetores = indice.gerar_embeddings(ModeloFalso(), documentos["texto"])
    idx = indice.IndiceVetorial(vetores, documentos["review_id"])
    padrao = busca.Buscador(documentos, idx, ModeloFalso())             # min_palavras = 3
    assert "r5" not in set(padrao.buscar("otimo", k=6, modo="vetorial")["review_id"])
    assert "r5" in set(padrao.buscar("otimo", k=6, modo="vetorial", min_palavras=1)["review_id"])
    # combina com os outros filtros: nota <= 1 deixa r0 e r2, mas r2 ("Veio quebrado") tem 2 palavras
    permitidos = busca.posicoes_permitidas(documentos, nota_max=1)
    assert set(padrao.buscar("x", k=6, modo="vetorial", permitidos=permitidos)["review_id"]) == {"r0"}
