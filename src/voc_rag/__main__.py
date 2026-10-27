"""Gera a base de conhecimento inteira, do download ao índice vetorial.

Uso, de qualquer pasta:  python -m voc_rag [--recalcular]

Cada etapa também roda sozinha (python -m voc_rag.dados, .preparacao, .indice);
o BM25 não tem etapa porque é montado em memória ao criar o Buscador.
"""
import sys
import time

from voc_rag import config, indice, preparacao
from voc_rag.dados import baixar_dados, carregar_tabelas


def main(recalcular=False):
    inicio = time.perf_counter()

    print("1/3 Dados brutos ...........", baixar_dados())

    documentos = preparacao.construir_base(carregar_tabelas())
    preparacao.salvar_base(documentos)
    print(f"2/3 Base de conhecimento .... {len(documentos):,} documentos em {config.ARQUIVO_BASE}")

    try:
        if recalcular:
            raise FileNotFoundError
        _, manifesto = indice.carregar_indice()
        print("3/3 Índice vetorial ........ já atualizado para esta base (use --recalcular para refazer)")
    except (FileNotFoundError, ValueError):
        idx, manifesto = indice.construir_indice(preparacao.carregar_base(), indice.carregar_modelo(),
                                                 mostrar_progresso=True)
        indice.salvar_indice(idx.vetores, idx.review_ids, manifesto)
        print(f"3/3 Índice vetorial ........ {manifesto['n_documentos']:,} vetores em "
              f"{config.PASTA_INDICE_VETORIAL} ({manifesto['segundos_embedding']} s de embeddings)")

    print(f"Pronto em {time.perf_counter() - inicio:.0f} s.")


if __name__ == "__main__":
    main(recalcular="--recalcular" in sys.argv)
