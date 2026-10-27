"""Recuperação: BM25, busca vetorial e a fusão híbrida com RRF (IA Gen aula 02).

- BM25 (lexical) acerta termos exatos e palavras raras.
- Vetorial (semântica) acerta paráfrases.
- RRF funde os dois rankings usando só as posições, não as pontuações, que
  estão em escalas diferentes.

O BM25 não é gravado em disco: montar sobre ~42 mil reviews curtas leva
poucos segundos, então ele é reconstruído a partir da base.
"""
import re
import time

import numpy as np
import pandas as pd

from voc_rag.config import MIN_PALAVRAS_EVIDENCIA, PROFUNDIDADE_FUSAO, RRF_K
from voc_rag.indice import gerar_embeddings
from voc_rag.texto import limpar_normalizar

MODOS = ("vetorial", "bm25", "hibrida")


# --- BM25 ---------------------------------------------------------------------------
def tokenizar_lexical(texto):
    """Tokenização simples para o BM25: minúsculas + palavras alfanuméricas (IA Gen aula 02)."""
    return re.findall(r"[a-z0-9À-ÿ\-]+", texto.lower())


class IndiceBM25:
    """BM25 sobre o `texto_lexical` (normalizado, sem acento) de cada documento."""

    def __init__(self, textos_lexicais):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenizar_lexical(t) for t in textos_lexicais])

    def buscar(self, consulta, k=10, permitidos=None):
        """Top-k como lista de (posição, pontuação BM25). Documentos sem nenhum termo ficam de fora.

        A consulta passa pela MESMA normalização da base (sem acento, contrações
        expandidas), senão "não" na pergunta não casaria com "nao" na review.
        """
        termos = tokenizar_lexical(limpar_normalizar(consulta, remove_acentos=True))
        pontuacoes = self.bm25.get_scores(termos)
        if permitidos is not None:
            mascara = np.zeros(len(pontuacoes), dtype=bool)
            mascara[np.asarray(permitidos, dtype="int64")] = True
            pontuacoes = np.where(mascara, pontuacoes, 0.0)
        ordem = np.argsort(-pontuacoes)[:k]
        return [(int(p), float(pontuacoes[p])) for p in ordem if pontuacoes[p] > 0]


# --- Fusão ---------------------------------------------------------------------------
def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: funde rankings somando 1/(k + posição) (IA Gen aula 02)."""
    pontos = {}
    for ranking in rankings:
        for posicao, idx in enumerate(ranking):
            pontos[idx] = pontos.get(idx, 0.0) + 1.0 / (k + posicao + 1)
    return sorted(pontos.items(), key=lambda par: -par[1])


# --- Filtros por metadado ---------------------------------------------------------------
def posicoes_permitidas(documentos, nota_min=None, nota_max=None, categorias=None, estados=None,
                        periodo=None, entregue_com_atraso=None):
    """Posições dos documentos que passam nos filtros (None = sem filtro naquele campo).

    `periodo` é um par ("2018-01", "2018-06") de ano_mes, inclusivo.
    """
    filtro = pd.Series(True, index=documentos.index)
    if nota_min is not None:
        filtro &= documentos["nota"] >= nota_min
    if nota_max is not None:
        filtro &= documentos["nota"] <= nota_max
    if categorias is not None:
        filtro &= documentos["categoria"].isin(categorias)
    if estados is not None:
        filtro &= documentos["estado"].isin(estados)
    if periodo is not None:
        filtro &= documentos["ano_mes"].between(*periodo)
    if entregue_com_atraso is not None:
        filtro &= documentos["entregue_com_atraso"].fillna(not entregue_com_atraso) == entregue_com_atraso
    return np.flatnonzero(filtro.to_numpy())


# --- Buscador -------------------------------------------------------------------------
class Buscador:
    """Junta base, índice vetorial, modelo de embeddings e BM25 numa interface só."""

    COLUNAS_RESULTADO = ["review_id", "texto", "nota", "categoria", "estado", "ano_mes",
                         "entregue_com_atraso"]

    def __init__(self, documentos, indice_vetorial, modelo, indice_bm25=None,
                 min_palavras=MIN_PALAVRAS_EVIDENCIA):
        if indice_vetorial.review_ids != documentos["review_id"].tolist():
            raise ValueError("índice vetorial e base estão fora de ordem")
        self.documentos = documentos.reset_index(drop=True)
        self.indice_vetorial = indice_vetorial
        self.modelo = modelo
        self.bm25 = indice_bm25 or IndiceBM25(self.documentos["texto_lexical"])
        self.min_palavras = min_palavras
        self._n_palavras = self.documentos["texto"].str.split().str.len().to_numpy()

    def _aplicar_min_palavras(self, permitidos, min_palavras):
        """Reviews curtas demais ("bom", "Certinho") não servem de evidência citada.

        Com pergunta curta, a busca vetorial as puxa com similaridade alta e sem
        sentido. Elas continuam na base para contagens; só saem da recuperação.
        """
        if not min_palavras or min_palavras <= 1:
            return permitidos
        elegiveis = np.flatnonzero(self._n_palavras >= min_palavras)
        if permitidos is None:
            return elegiveis
        return np.intersect1d(np.asarray(permitidos, dtype="int64"), elegiveis)

    def buscar(self, consulta, k=10, modo="hibrida", permitidos=None, profundidade=PROFUNDIDADE_FUSAO,
               min_palavras=None):
        """Top-k como DataFrame: posição em cada ranking, pontuações e metadados para citar.

        Na híbrida, cada busca traz `profundidade` candidatos e o RRF reordena.
        `similaridade` (cosseno da vetorial) vai em todos os modos: é o sinal de
        evidência que o RAG vai usar para recusar. `min_palavras` (padrão: o do
        Buscador) tira da busca as reviews curtas demais; use 1 para desligar.
        """
        if modo not in MODOS:
            raise ValueError(f"modo deve ser um de {MODOS}")
        permitidos = self._aplicar_min_palavras(
            permitidos, self.min_palavras if min_palavras is None else min_palavras)
        inicio = time.perf_counter()
        profundidade = max(profundidade, k)
        vetor = gerar_embeddings(self.modelo, [consulta])[0]   # uma vez só, serve aos dois usos
        vetorial = (self.indice_vetorial.buscar(vetor, profundidade, permitidos)
                    if modo != "bm25" else [])
        lexical = self.bm25.buscar(consulta, profundidade, permitidos) if modo != "vetorial" else []

        if modo == "vetorial":
            ordem = vetorial[:k]
        elif modo == "bm25":
            ordem = lexical[:k]
        else:
            ordem = rrf([[p for p, _ in vetorial], [p for p, _ in lexical]])[:k]

        rank_vet = {p: i + 1 for i, (p, _) in enumerate(vetorial)}
        rank_lex = {p: i + 1 for i, (p, _) in enumerate(lexical)}
        pont_lex = dict(lexical)
        posicoes = [p for p, _ in ordem]
        similaridades = self.indice_vetorial.vetores[posicoes] @ vetor if posicoes else np.array([])

        resultado = self.documentos.iloc[posicoes][self.COLUNAS_RESULTADO].copy()
        resultado.insert(0, "pontuacao", [round(s, 4) for _, s in ordem])
        resultado.insert(1, "similaridade", similaridades.round(3))
        resultado.insert(2, "pos_vetorial", pd.array([rank_vet.get(p) for p in posicoes], dtype="Int64"))
        resultado.insert(3, "pos_bm25", pd.array([rank_lex.get(p) for p in posicoes], dtype="Int64"))
        resultado["bm25"] = [round(pont_lex.get(p, 0.0), 2) for p in posicoes]
        resultado.attrs["latencia_ms"] = (time.perf_counter() - inicio) * 1000
        resultado.attrs["modo"] = modo
        resultado.index = pd.RangeIndex(1, len(resultado) + 1, name="rank")
        return resultado
