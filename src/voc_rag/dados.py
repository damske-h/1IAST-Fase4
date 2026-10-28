"""Obtenção e leitura dos dados brutos da Olist (Kaggle)."""
import shutil
from pathlib import Path

import pandas as pd

from voc_rag.config import ARQUIVOS, DATASET_KAGGLE, PASTA_BRUTOS


def arquivos_faltando(pasta=PASTA_BRUTOS):
    """Nomes dos CSVs que ainda não estão na pasta."""
    return [nome for nome in ARQUIVOS.values() if not (Path(pasta) / nome).exists()]


def baixar_dados(pasta=PASTA_BRUTOS, forcar=False):
    """Garante os 6 CSVs em data/raw (baixa com kagglehub se faltar) e devolve a pasta."""
    pasta = Path(pasta)
    if not arquivos_faltando(pasta) and not forcar:
        return pasta
    import kagglehub

    pasta.mkdir(parents=True, exist_ok=True)
    try:
        # output_dir grava direto no repo; sem ele o kagglehub usa o cache
        # do usuário (~/.cache/kagglehub/...), fora de qualquer repositório
        origem = kagglehub.dataset_download(DATASET_KAGGLE, output_dir=str(pasta),
                                            force_download=forcar)
    except (TypeError, FileExistsError):
        # TypeError: kagglehub antigo, sem output_dir. FileExistsError: o kagglehub 1.x
        # recusa uma pasta não vazia (data/raw tem o .gitkeep). Nos dois casos baixamos
        # no cache e copiamos os CSVs.
        origem = kagglehub.dataset_download(DATASET_KAGGLE, force_download=forcar)

    for nome in arquivos_faltando(pasta):
        achados = sorted(Path(origem).rglob(nome))
        if achados:
            shutil.copy2(achados[0], pasta / nome)

    if arquivos_faltando(pasta):
        raise FileNotFoundError(f"Ainda faltam em {pasta}: {arquivos_faltando(pasta)}")
    return pasta


def carregar_tabelas(pasta=PASTA_BRUTOS):
    """Lê os 6 CSVs e devolve um dicionário {apelido: DataFrame}."""
    return {chave: pd.read_csv(Path(pasta) / nome) for chave, nome in ARQUIVOS.items()}


if __name__ == "__main__":
    # Uso, de qualquer pasta:  python -m voc_rag.dados [--forcar]
    import sys

    print(f"CSVs da Olist em: {baixar_dados(forcar='--forcar' in sys.argv)}")
