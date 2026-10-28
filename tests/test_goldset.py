import pandas as pd
import pytest

from voc_rag import goldset
from voc_rag.texto import limpar_normalizar


def test_gold_set_versionado_e_valido():
    itens = goldset.carregar_gold()
    assert {i.estrato for i in itens} >= set("ADEFG")
    assert all(i.esperado == "respondida" for i in itens if i.estrato in "ABCD")
    assert all(i.esperado == "recusada" for i in itens if i.estrato in "EF")
    assert all(i.subtemas for i in itens if i.estrato == "A")
    assert all(i.filtros.como_kwargs() for i in itens if i.estrato == "D")
    assert {"dev", "teste"} == {i.particao for i in itens}


def test_carregar_gold_recusa_b_sem_alvo(tmp_path):
    caminho = tmp_path / "g.csv"
    pd.DataFrame([{"id": "B01", "estrato": "B", "particao": "dev", "pergunta": "Qual a voltagem?",
                   "esperado": "respondida", "filtros": "", "subtemas": "", "review_alvo": "",
                   "observacao": ""}]).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="sem review_alvo"):
        goldset.carregar_gold(caminho)


def test_particionar_estratificado_e_reprodutivel():
    estratos = list("A" * 12 + "F" * 6)
    p1 = goldset.particionar(estratos)
    assert p1 == goldset.particionar(estratos)
    assert p1[:12].count("teste") == 4 and p1[12:].count("teste") == 2


def test_candidatos_agulha_pega_termos_raros():
    textos = [f"produto chegou bem e funcionou direitinho como esperado numero {i}" for i in range(30)]
    textos += ["comprei o liquidificador e a voltagem veio errada, era bivolt no anuncio mas chegou 220"]
    docs = pd.DataFrame({"review_id": [f"r{i}" for i in range(31)], "texto": textos,
                         "texto_lexical": [limpar_normalizar(t, remove_acentos=True) for t in textos],
                         "nota": [5] * 30 + [1], "categoria": ["x"] * 31, "estado": ["SP"] * 31,
                         "ano_mes": ["2018-01"] * 31})
    cand = goldset.candidatos_agulha(docs, max_df=1)
    assert "r30" in set(cand["review_id"])
    assert "voltagem" in cand.set_index("review_id").loc["r30", "termos_raros"]


@pytest.mark.parametrize("texto, esperado", [
    ("A maioria dos clientes reclama do frete", True), ("70% das reviews citam atraso", True),
    ("Nas reviews recuperadas, clientes relatam atraso", False)])
def test_contem_quantificacao(texto, esperado):
    assert goldset.contem_quantificacao(texto) is esperado
