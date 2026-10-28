import pandas as pd
import pytest

from tests.llm_falso import LLMFalso
from tests.modelo_falso import ModeloFalso
from voc_rag import busca, demo, indice, rag
from voc_rag.schemas import Evidencia, RespostaRAG
from voc_rag.texto import limpar_normalizar

OPCOES = {"categorias": ["a", "b"], "estados": ["RJ", "SP"], "meses": ["2017-01", "2018-08"]}


def test_montar_filtros_so_com_o_que_mudou():
    assert demo.montar_filtros(OPCOES) is None
    assert demo.montar_filtros(OPCOES, periodo=("2017-01", "2018-08")) is None
    filtros = demo.montar_filtros(OPCOES, ["SP"], ["a"], (1, 2), ("2018-01", "2018-06"), "só com atraso")
    assert filtros == {"estados": ["SP"], "categorias": ["a"], "nota_max": 2,
                       "periodo": ["2018-01", "2018-06"], "entregue_com_atraso": True}


def test_tabelas_da_resposta():
    r = RespostaRAG(status="respondida", resumo="x", evidencias=[
        Evidencia(review_id="r1", texto="chegou", nota=5, categoria="a", estado="SP", ano_mes="2018-01")])
    assert demo.tabela_evidencias(r).iloc[0]["review_id"] == "r1"
    p = demo.passos(["guardrail -> ok", "rotear -> analise_reviews {}", "auditar"])
    assert list(p["etapa"]) == ["guardrail", "rotear", "auditar"] and p.iloc[2]["resultado"] == ""


def test_resultados_da_avaliacao_vem_do_cache():
    tabelas = demo.resultados_avaliacao()
    assert list(tabelas["teste"].index) == ["B0", "H5"]
    assert tabelas["teste"].loc["H5", "comportamento certo"] >= tabelas["teste"].loc["B0", "comportamento certo"]


@pytest.fixture
def recursos_falsos(monkeypatch):
    textos = ["Meu pedido não chegou até hoje", "O pedido nunca chegou na minha casa",
              "Chegou rápido, recomendo muito"]
    documentos = pd.DataFrame({
        "review_id": ["r0", "r1", "r2"], "texto": textos,
        "texto_lexical": [limpar_normalizar(t, remove_acentos=True) for t in textos],
        "nota": [1, 1, 5], "categoria": ["a", "a", "b"], "estado": ["SP", "RJ", "SP"],
        "ano_mes": ["2018-01", "2018-02", "2018-03"],
        "entregue_com_atraso": pd.array([pd.NA] * 3, dtype="boolean")})
    vetores = indice.gerar_embeddings(ModeloFalso(), documentos["texto"])
    buscador = busca.Buscador(documentos, indice.IndiceVetorial(vetores, documentos["review_id"]), ModeloFalso())
    llm = LLMFalso({"Roteamento": {"intencao": "analise_reviews", "consulta": "pedido não chegou"},
                    "Variacoes": {"consultas": ["não recebi"]},
                    "JulgamentoEvidencia": {"relevantes": ["r0", "r1"], "motivo": "não entrega"},
                    "RespostaGerada": {"resumo": "Pedidos não chegaram.",
                                       "pontos": [{"afirmacao": "Não chegou", "review_ids": ["r0"]}]}})
    app = rag.criar_rag(documentos, buscador, llm, limiar=0.0, caminho_log=None)
    monkeypatch.setattr(demo, "carregar_recursos", lambda: (documentos, app))


def test_app_responde_e_mostra_a_avaliacao(recursos_falsos):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception
    at.text_area(key="pergunta").set_value("O que dizem dos pedidos que não chegaram?")
    at.button[-1].click().run()                       # o botão do formulário é o último
    assert not at.exception
    assert any("RESPONDIDA" in s.value for s in at.success)
    assert any("comportamento certo" in str(d.value.columns.tolist()) for d in at.dataframe)
