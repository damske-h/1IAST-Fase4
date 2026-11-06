"""Voice of Customer Intelligence com RAG — Tech Challenge Fase 4 (FIAP IAST)."""
import os
import sys

# No macOS, o PyTorch e o FAISS trazem cada um sua cópia do OpenMP; com várias threads,
# a primeira busca no FAISS derruba o processo (segfault). Uma thread resolve e não pesa:
# a busca exata em ~42 mil vetores leva milissegundos. Precisa valer antes de importar torch.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
