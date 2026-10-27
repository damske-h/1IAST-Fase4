import sys

import numpy as np
import pandas as pd
import pytest

from tests.modelo_falso import ModeloFalso
from voc_rag import indice


@pytest.fixture
def documentos():
    return pd.DataFrame({"review_id": ["r1", "r2", "r3", "r4"],
                         "texto": ["nao chegou", "chegou rapido", "veio quebrado", "nao chegou ainda"],
                         "nota": [1, 5, 1, 2]})


@pytest.fixture
def base_csv(documentos, tmp_path):
    caminho = tmp_path / "base.csv"
    documentos.to_csv(caminho, index=False)
    return caminho


def test_embeddings_normalizados(documentos):
    vetores = indice.gerar_embeddings(ModeloFalso(), documentos["texto"])
    assert vetores.dtype == np.float32 and vetores.shape == (4, 26)
    assert np.allclose(np.linalg.norm(vetores, axis=1), 1)


def test_busca_acha_o_proprio_texto_primeiro(documentos, base_csv):
    idx, _ = indice.construir_indice(documentos, ModeloFalso(), "falso", base_csv)
    resultado = indice.buscar_texto(idx, ModeloFalso(), "veio quebrado", k=2)
    assert resultado[0][0] == 2 and resultado[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(resultado) == 2


def test_filtro_por_metadado_restringe_antes_de_ranquear(documentos, base_csv):
    idx, _ = indice.construir_indice(documentos, ModeloFalso(), "falso", base_csv)
    permitidos = np.flatnonzero(documentos["nota"] >= 2)          # r2 e r4
    resultado = indice.buscar_texto(idx, ModeloFalso(), "nao chegou", k=10, permitidos=permitidos)
    assert [p for p, _ in resultado] == [3, 1]
    assert indice.buscar_texto(idx, ModeloFalso(), "x", permitidos=[]) == []


def test_numpy_e_faiss_dao_o_mesmo_ranking(documentos, base_csv, monkeypatch):
    com_faiss, _ = indice.construir_indice(documentos, ModeloFalso(), "falso", base_csv)
    monkeypatch.setitem(sys.modules, "faiss", None)               # simula faiss não instalado
    sem_faiss, _ = indice.construir_indice(documentos, ModeloFalso(), "falso", base_csv)
    assert sem_faiss.motor == "numpy"
    for permitidos in (None, [0, 3]):
        a = indice.buscar_texto(com_faiss, ModeloFalso(), "chegou", k=3, permitidos=permitidos)
        b = indice.buscar_texto(sem_faiss, ModeloFalso(), "chegou", k=3, permitidos=permitidos)
        assert [p for p, _ in a] == [p for p, _ in b]


def test_salvar_carregar_e_detectar_base_desatualizada(documentos, base_csv, tmp_path):
    idx, manifesto = indice.construir_indice(documentos, ModeloFalso(), "falso", base_csv)
    assert manifesto["n_documentos"] == 4 and manifesto["dimensao"] == 26
    pasta = indice.salvar_indice(idx.vetores, idx.review_ids, manifesto, tmp_path / "idx")

    relido, _ = indice.carregar_indice(pasta, base_csv, nome_modelo="falso")
    assert relido.review_ids == ["r1", "r2", "r3", "r4"]
    assert np.array_equal(relido.vetores, idx.vetores)

    with pytest.raises(ValueError, match="foi gerado com falso"):
        indice.carregar_indice(pasta, base_csv, nome_modelo="outro-modelo")

    base_csv.write_text("review_id,texto\nr9,outra base\n")
    with pytest.raises(ValueError, match="outra versão da base"):
        indice.carregar_indice(pasta, base_csv, nome_modelo="falso")
