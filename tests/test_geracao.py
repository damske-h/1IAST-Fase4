import pandas as pd
import pytest

from tests.llm_falso import LLMFalso
from voc_rag import geracao, guardrails
from voc_rag.schemas import RespostaGerada


@pytest.fixture
def evidencias():
    return pd.DataFrame({
        "review_id": ["r1", "r2", "r3"],
        "texto": ["Não chegou até hoje", "Pedido nunca foi entregue", "Ainda aguardo o produto"],
        "nota": [1, 1, 2], "categoria": ["a"] * 3, "estado": ["SP"] * 3, "ano_mes": ["2018-01"] * 3,
    })


def test_validar_citacoes_remove_ids_e_pontos_sem_fonte():
    bruta = RespostaGerada(resumo="x", pontos=[
        {"afirmacao": "Pedidos não chegaram", "review_ids": ["r1", "r9", "r1", "r2"]},
        {"afirmacao": "Frete caro", "review_ids": ["r99"]},
    ])
    pontos, avisos = geracao.validar_citacoes(bruta, ["r1", "r2", "r3"])
    assert [p.review_ids for p in pontos] == [["r1", "r2"]]
    assert len(avisos) == 3            # r9, r99 e o ponto descartado


def test_gerar_resposta_mascara_pii_e_usa_contexto(evidencias):
    llm = LLMFalso({"resumo": "Clientes não receberam; um deixou o contato ana@x.com",
                    "pontos": [{"afirmacao": "Ligue (11) 98877-1234", "review_ids": ["r2"]}]})
    resumo, pontos, avisos = geracao.gerar_resposta(geracao.criar_gerador(llm), "Pedido chegou?", evidencias)
    assert "[EMAIL]" in resumo and "[TELEFONE]" in pontos[0].afirmacao
    assert any("PII mascarada" in a for a in avisos)
    assert 'id="r3"' in llm.prompts[0] and "são 3 reviews" in llm.prompts[0]


def test_gerar_resposta_sem_citacao_valida_devolve_pontos_vazios(evidencias):
    llm = LLMFalso({"resumo": "x", "pontos": [{"afirmacao": "y", "review_ids": ["inventado"]}]})
    _, pontos, _ = geracao.gerar_resposta(geracao.criar_gerador(llm), "p", evidencias)
    assert pontos == []


def test_schema_exige_citacao():
    with pytest.raises(Exception):
        RespostaGerada(resumo="x", pontos=[{"afirmacao": "y", "review_ids": []}])


def test_mascarar_pii_saida():
    assert guardrails.mascarar_pii_saida("CPF 529.982.247-25 ok") == ("CPF [CPF] ok", ["CPF"])
