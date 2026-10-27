from voc_rag.texto import limpar_leve, limpar_normalizar, mascarar_pii


def test_limpar_leve_tira_quebras_e_mantem_caixa_acentos_e_pontuacao():
    assert limpar_leve("Não  chegou!\r\nPéssimo\\r\\nmesmo") == "Não chegou! Péssimo mesmo"


def test_limpar_normalizar_expande_contracoes_e_tira_acentos():
    assert limpar_normalizar("N chegou, vc viu?", remove_acentos=True) == "nao chegou, voce viu?"
    assert limpar_normalizar(None) == ""


def test_mascarar_pii():
    texto, achados = mascarar_pii("Ligue (11) 98877-1234 ou mande para ana@x.com")
    assert texto == "Ligue [TELEFONE] ou mande para [EMAIL]"
    assert sorted(achados) == ["EMAIL", "TELEFONE"]


def test_mascarar_pii_nao_mexe_em_texto_limpo():
    assert mascarar_pii("Chegou em 3 dias, nota 10") == ("Chegou em 3 dias, nota 10", [])
