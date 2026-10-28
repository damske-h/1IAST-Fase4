import json

import pandas as pd
import pytest

from tests.llm_falso import LLMFalso
from voc_rag import experimento as ex
from voc_rag.goldset import ItemGold
from voc_rag.schemas import Ponto


def _item(id_, esperado="respondida", **extra):
    return ItemGold(id=id_, estrato=id_[0], particao="dev", pergunta=f"pergunta {id_}",
                    esperado=esperado, **extra)


ITENS = [_item("A01", subtemas="atraso|defeito"), _item("B01", review_alvo="r2"),
         _item("D01", filtros='{"estados": ["SP"], "nota_max": 2}'),
         _item("E01", "recusada"), _item("G01", "bloqueada")]


def _resp(sistema, id_, status="respondida", recuperados=("r1", "r2", "r3"), citados=("r1",), **extra):
    pontos = [{"afirmacao": "Chegou atrasado", "review_ids": list(citados)}] if status == "respondida" else []
    return {"sistema": sistema, "id": id_, "status": status, "resumo": extra.pop("resumo", "ok"),
            "pontos": pontos, "recuperados": list(recuperados), "citados": list(citados) if pontos else [],
            "consultas": [], "filtros": extra.pop("filtros", {}), "avisos": [], "chamadas": 1,
            "tokens_entrada": 10, "tokens_saida": 5, "tokens_raciocinio": 0, "latencia_ms": 1000}


def test_teste_trancado_e_pastas():
    with pytest.raises(ValueError, match="trancado"):
        ex.ConfigAvaliacao(particao="teste")
    assert ex.ConfigAvaliacao().pasta_saida == ex.PASTA_RESPOSTAS
    assert ex.ConfigAvaliacao(modo="ao_vivo").pasta_saida == ex.PASTA_AO_VIVO


def test_max_perguntas_alterna_estratos():
    itens = ex.selecionar_perguntas(ex.ConfigAvaliacao(max_perguntas=7))
    assert len(itens) == 7 and len({i.estrato for i in itens}) == 7


def test_estimativa_soma_sistemas_e_juizes():
    config = ex.ConfigAvaliacao(sistemas=["B0", "H4"], juiz_fidelidade=True, juiz_relevancia=True)
    tabela = ex.estimar_chamadas(config, ITENS)
    assert tabela.iloc[-1]["chamadas (máx.)"] == 5 * 1 + 5 * 4 + 3 + 5 * 2


def test_cache_continua_de_onde_parou(tmp_path):
    config = ex.ConfigAvaliacao(sistemas=["B0"], modo="ao_vivo", pasta=tmp_path)
    chamadas = []

    def b0(pergunta):
        chamadas.append(pergunta)
        return {"status": "respondida", "resumo": "x", "pontos": [Ponto(afirmacao="a", review_ids=["r1"])],
                "citados": ["r1"], "recuperados": ["r1"]}
    ex.rodar_sistemas(config, {"B0": b0}, ITENS[:2])
    ex.rodar_sistemas(config, {"B0": b0}, ITENS[:3])
    assert len(chamadas) == 3
    respostas = ex.ler_respostas(config)
    assert list(respostas["id"]) == ["A01", "B01", "D01"] and respostas["citados"][0] == ["r1"]
    with pytest.raises(ValueError, match="modo cache"):
        ex.rodar_sistemas(ex.ConfigAvaliacao(), {}, ITENS)


def test_metricas_por_pergunta():
    respostas = pd.DataFrame([
        _resp("B0", "A01", citados=("r1", "r9")), _resp("B0", "B01"),
        _resp("B0", "D01", filtros={"nota_max": 2, "estados": ["SP"]}),
        _resp("B0", "E01", resumo="A maioria dos clientes reclama"), _resp("B0", "G01", "recusada"),
    ])
    julgamentos = pd.DataFrame({"pergunta_id": ["A01", "A01", "A01"], "review_id": ["r1", "r2", "r3"],
                                "relevante": [True, True, False], "subtema": ["atraso", "defeito", "outro"]})
    m = ex.metricas_por_pergunta(respostas, ITENS, julgamentos).set_index("id")
    assert m.loc["A01", "precisao@10"] == pytest.approx(2 / 3)
    assert m.loc["A01", "cobertura_subtemas"] == 0.5 and m.loc["A01", "citacoes_validas"] == 0.5
    assert m.loc["B01", "acerto@10"] and m.loc["B01", "rr"] == 0.5
    assert m.loc["D01", "filtros_exatos"]
    assert not m.loc["E01", "acerto_comportamento"] and m.loc["E01", "quantificacao_sem_base"]
    assert m.loc["G01", "acerto_comportamento"] and not m.loc["G01", "bloqueou_no_guardrail"]

    recusa = ex.matriz_recusa(m.reset_index()).loc["B0"]
    assert (recusa["recusa correta"], recusa["alucinação"], recusa["falsa recusa"]) == (1, 1, 0)
    resumo = ex.resumo_por_sistema(m.reset_index())
    assert resumo.loc["B0", "MRR"] == 0.5 and resumo.loc["B0", "chamadas"] == 5


def test_comparacao_pareada_e_bootstrap():
    m = pd.DataFrame({"sistema": ["B0"] * 3 + ["H1"] * 3, "id": ["a", "b", "c"] * 2,
                      "acerto": [0, 1, 1, 1, 1, 0]})
    assert ex.comparacao_pareada(m, "acerto", "B0", "H1") == {"vitórias": 1, "empates": 1, "derrotas": 1}
    ic = ex.bootstrap_ic(m, "acerto")
    assert ic.loc["B0", "IC 2,5%"] <= ic.loc["B0", "média"] <= ic.loc["B0", "IC 97,5%"]


@pytest.fixture
def documentos():
    return pd.DataFrame({"review_id": ["r1", "r2", "r3"], "texto": ["atrasou", "quebrado", "ótimo"],
                         "nota": [1, 1, 5], "categoria": ["a"] * 3, "estado": ["SP"] * 3,
                         "ano_mes": ["2018-01"] * 3})


def test_juizes_gravam_e_nao_repetem(tmp_path, documentos, monkeypatch):
    monkeypatch.setattr(ex, "PASTA_RESPOSTAS", tmp_path / "versionado")   # isola do cache real
    config = ex.ConfigAvaliacao(sistemas=["B0"], modo="ao_vivo", pasta=tmp_path)
    respostas = pd.DataFrame([_resp("B0", "A01"), _resp("B0", "E01", "recusada")])
    llm = LLMFalso({"JulgamentoRelevancia": {"rotulos": [
                        {"review_id": "r1", "relevante": True, "subtema": "atraso"},
                        {"review_id": "r2", "relevante": False, "subtema": "outro"}]},
                    "JulgamentoFidelidade": {"veredictos": [{"indice": 1, "sustentada": True, "motivo": "ok"}]}})
    ex.rotular_pool(config, ITENS, respostas, documentos, llm)
    ex.julgar_fidelidade(config, ITENS, respostas, documentos, llm)
    ex.rotular_pool(config, ITENS, respostas, documentos, llm)
    ex.julgar_fidelidade(config, ITENS, respostas, documentos, llm)
    assert len(llm.prompts) == 2                                  # 1 pool (só A01) + 1 fidelidade
    julg = pd.read_csv(tmp_path / "julgamentos.csv").set_index("review_id")
    assert julg.loc["r3", "subtema"] == "sem rótulo" and not julg.loc["r3", "relevante"]
    fid = ex.ler_fidelidade(config)
    assert fid.loc[0, "fidelidade"] == 1.0 and json.loads(fid.loc[0, "veredictos"])[0]["sustentada"]
