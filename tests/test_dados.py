from pathlib import Path

import pandas as pd

from voc_rag import config, dados


def test_caminhos_partem_da_raiz_do_repo():
    assert (config.RAIZ / "pyproject.toml").exists()
    assert config.PASTA_BRUTOS == config.RAIZ / "data" / "raw"


def test_arquivos_faltando_e_carregar_tabelas(tmp_path):
    assert len(dados.arquivos_faltando(tmp_path)) == len(config.ARQUIVOS)
    for nome in config.ARQUIVOS.values():
        pd.DataFrame({"a": [1]}).to_csv(tmp_path / nome, index=False)
    assert dados.arquivos_faltando(tmp_path) == []
    tabelas = dados.carregar_tabelas(tmp_path)
    assert set(tabelas) == set(config.ARQUIVOS)


def test_baixar_copia_do_cache_quando_kagglehub_nao_tem_output_dir(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "versions" / "1"
    cache.mkdir(parents=True)
    for nome in config.ARQUIVOS.values():
        (cache / nome).write_text("a\n1\n")

    class KagglehubAntigo:
        @staticmethod
        def dataset_download(handle, path=None, *, force_download=False):
            return str(tmp_path / "cache")

    monkeypatch.setitem(__import__("sys").modules, "kagglehub", KagglehubAntigo)
    destino = tmp_path / "raw"
    assert dados.baixar_dados(destino) == destino
    assert dados.arquivos_faltando(destino) == []
    assert Path(destino / config.ARQUIVOS["reviews"]).read_text() == "a\n1\n"
