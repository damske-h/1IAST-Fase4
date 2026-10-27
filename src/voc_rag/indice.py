"""Índice vetorial da base de conhecimento: embeddings + FAISS (IA Gen aula 02).

O caro é gerar os embeddings; montar um índice FAISS *flat* com 40 mil
vetores leva milissegundos. Por isso gravamos só os embeddings (.npy), a
ordem dos review_id e um manifesto, e o índice é remontado ao carregar.

Para gerar o índice sem abrir o notebook:
    python -m voc_rag.indice
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from voc_rag.config import ARQUIVO_BASE, MODELO_EMBEDDINGS, PASTA_INDICE_VETORIAL, TAMANHO_LOTE

ARQUIVO_EMBEDDINGS = "embeddings.npy"
ARQUIVO_IDS = "review_ids.txt"
ARQUIVO_MANIFESTO = "manifesto.json"


# --- Embeddings ---------------------------------------------------------------------
def carregar_modelo(nome=MODELO_EMBEDDINGS):
    """Carrega o modelo de embeddings (baixa do Hugging Face na primeira vez)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nome)


def gerar_embeddings(modelo, textos, tamanho_lote=TAMANHO_LOTE, mostrar_progresso=False):
    """Uma matriz (n_textos x dimensão), float32, com cada linha de norma 1.

    Normalizar faz o produto interno virar similaridade do cosseno (PLN aula 04).
    """
    vetores = modelo.encode(list(textos), batch_size=tamanho_lote, normalize_embeddings=True,
                            show_progress_bar=mostrar_progresso, convert_to_numpy=True)
    return np.asarray(vetores, dtype="float32")


# --- Índice -------------------------------------------------------------------------
class IndiceVetorial:
    """Busca exata por similaridade do cosseno, com FAISS ou, sem ele, com numpy."""

    def __init__(self, vetores, review_ids):
        if len(vetores) != len(review_ids):
            raise ValueError("cada vetor precisa de um review_id")
        self.vetores = np.ascontiguousarray(vetores, dtype="float32")
        self.review_ids = list(review_ids)
        try:
            import faiss

            self._faiss = faiss
            self.faiss = faiss.IndexFlatIP(self.vetores.shape[1])  # produto interno = cosseno
            self.faiss.add(self.vetores)
        except ImportError:  # mesmo resultado, só mais lento em bases grandes
            self.faiss = None

    @property
    def motor(self):
        return "faiss" if self.faiss is not None else "numpy"

    def buscar(self, vetor_consulta, k=10, permitidos=None):
        """Top-k como lista de (posição, similaridade), da mais parecida para a menos.

        `permitidos` (posições na base) restringe a busca ANTES de ranquear: é o
        filtro por metadado ("só notas 1 e 2"). Sem ele, o top-k poderia vir todo
        de reviews que o filtro depois descartaria.
        """
        consulta = np.asarray(vetor_consulta, dtype="float32").reshape(1, -1)
        if permitidos is not None:
            permitidos = np.asarray(permitidos, dtype="int64")
            if len(permitidos) == 0:
                return []
        if self.faiss is not None:
            params = None
            if permitidos is not None:
                seletor = self._faiss.IDSelectorBatch(permitidos)
                params = self._faiss.SearchParameters(sel=seletor)
            notas, posicoes = self.faiss.search(consulta, k, params=params)
            return [(int(p), float(s)) for p, s in zip(posicoes[0], notas[0]) if p != -1]

        candidatos = np.arange(len(self.vetores)) if permitidos is None else permitidos
        similaridades = self.vetores[candidatos] @ consulta[0]
        ordem = np.argsort(-similaridades)[:k]
        return [(int(candidatos[i]), float(similaridades[i])) for i in ordem]


def buscar_texto(indice, modelo, consulta, k=10, permitidos=None):
    """Atalho: gera o embedding da pergunta e busca no índice."""
    return indice.buscar(gerar_embeddings(modelo, [consulta])[0], k=k, permitidos=permitidos)


# --- Reprodutibilidade --------------------------------------------------------------
def hash_arquivo(caminho):
    """SHA-256 do arquivo: identifica qual versão da base gerou o índice."""
    h = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _versao(pacote):
    try:
        return version(pacote)
    except PackageNotFoundError:
        return None


def montar_manifesto(vetores, nome_modelo, caminho_base, segundos):
    """O que é preciso saber para reproduzir (ou desconfiar de) um índice."""
    return {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": nome_modelo,
        "n_documentos": int(vetores.shape[0]),
        "dimensao": int(vetores.shape[1]),
        "normalizado": True,
        "metrica": "cosseno (produto interno de vetores normalizados)",
        "base": str(Path(caminho_base).name),
        "base_sha256": hash_arquivo(caminho_base),
        "segundos_embedding": round(segundos, 1),
        "versoes": {p: _versao(p) for p in ["sentence-transformers", "torch", "faiss-cpu", "numpy"]},
    }


def salvar_indice(vetores, review_ids, manifesto, pasta=PASTA_INDICE_VETORIAL):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    np.save(pasta / ARQUIVO_EMBEDDINGS, vetores)
    (pasta / ARQUIVO_IDS).write_text("\n".join(review_ids), encoding="utf-8")
    (pasta / ARQUIVO_MANIFESTO).write_text(json.dumps(manifesto, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return pasta


def carregar_indice(pasta=PASTA_INDICE_VETORIAL, caminho_base=ARQUIVO_BASE,
                    nome_modelo=MODELO_EMBEDDINGS):
    """Remonta o índice a partir do disco e confere se ele bate com a base e o modelo atuais."""
    pasta = Path(pasta)
    manifesto = json.loads((pasta / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))
    if nome_modelo is not None and manifesto["modelo"] != nome_modelo:
        # pergunta e reviews precisam estar no MESMO espaço vetorial
        raise ValueError(f"o índice foi gerado com {manifesto['modelo']}, não com {nome_modelo}")
    if caminho_base is not None and manifesto["base_sha256"] != hash_arquivo(caminho_base):
        raise ValueError("o índice foi gerado com outra versão da base: rode python -m voc_rag.indice")
    vetores = np.load(pasta / ARQUIVO_EMBEDDINGS)
    review_ids = (pasta / ARQUIVO_IDS).read_text(encoding="utf-8").split("\n")
    return IndiceVetorial(vetores, review_ids), manifesto


def construir_indice(documentos, modelo, nome_modelo=MODELO_EMBEDDINGS, caminho_base=ARQUIVO_BASE,
                     mostrar_progresso=False):
    """Embeddings do `texto` de cada documento + índice + manifesto."""
    inicio = time.perf_counter()
    vetores = gerar_embeddings(modelo, documentos["texto"], mostrar_progresso=mostrar_progresso)
    manifesto = montar_manifesto(vetores, nome_modelo, caminho_base, time.perf_counter() - inicio)
    return IndiceVetorial(vetores, documentos["review_id"]), manifesto


if __name__ == "__main__":
    from voc_rag.preparacao import carregar_base

    documentos = carregar_base()
    print(f"Gerando embeddings de {len(documentos):,} reviews com {MODELO_EMBEDDINGS}...")
    indice, manifesto = construir_indice(documentos, carregar_modelo(), mostrar_progresso=True)
    pasta = salvar_indice(indice.vetores, indice.review_ids, manifesto)
    print(f"Índice ({indice.motor}) gravado em {pasta} em {manifesto['segundos_embedding']} s")
