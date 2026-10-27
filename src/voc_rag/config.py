"""Configuração central do projeto: caminhos e parâmetros num lugar só.

Notebooks, scripts e testes importam daqui. Assim, trocar o modelo de
embeddings ou a pasta de saída é mudar uma linha, não caçar strings soltas.
"""
from pathlib import Path

# --- Caminhos (sempre a partir da raiz do repositório, nunca do cwd) ---------
RAIZ = Path(__file__).resolve().parents[2]       # src/voc_rag/config.py -> raiz
PASTA_DADOS = RAIZ / "data"
PASTA_BRUTOS = PASTA_DADOS / "raw"               # CSVs do Kaggle, intocados
PASTA_INTERMEDIARIOS = PASTA_DADOS / "interim"   # etapas intermediárias, descartáveis
PASTA_PROCESSADOS = PASTA_DADOS / "processed"    # base de conhecimento pronta
PASTA_ARTEFATOS = RAIZ / "artifacts"             # índices FAISS/BM25 e manifestos

ARQUIVO_BASE = PASTA_PROCESSADOS / "base_conhecimento_reviews.csv"
PASTA_INDICE_VETORIAL = PASTA_ARTEFATOS / "indice_vetorial"  # embeddings + manifesto

# --- Fonte dos dados ----------------------------------------------------------
DATASET_KAGGLE = "olistbr/brazilian-ecommerce"
ARQUIVOS = {
    "reviews": "olist_order_reviews_dataset.csv",
    "pedidos": "olist_orders_dataset.csv",
    "clientes": "olist_customers_dataset.csv",
    "itens": "olist_order_items_dataset.csv",
    "produtos": "olist_products_dataset.csv",
    "traducao": "product_category_name_translation.csv",
}

# --- Parâmetros do pipeline (IA Gen aula 02) ----------------------------------
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIMITE_CHUNK = 400      # caracteres; reviews acima disso seriam divididas
TAMANHO_LOTE = 64       # textos por lote no encode (memória x velocidade)
RRF_K = 60              # constante do RRF; 60 é o valor do artigo original e da aula 02
PROFUNDIDADE_FUSAO = 50 # candidatos de cada busca que entram na fusão híbrida
MIN_PALAVRAS_EVIDENCIA = 3  # reviews com menos palavras não servem de evidência citada
LIMIAR_EVIDENCIA = 0.55 # similaridade mínima do top-1 para haver evidência (provisório, Etapa 3)
SEMENTE = 42            # amostras e exemplos reprodutíveis
