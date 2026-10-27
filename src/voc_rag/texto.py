"""Limpeza de texto e mascaramento de PII das reviews.

Duas limpezas, porque a limpeza é uma decisão sensível ao downstream
(PLN aula 02, slide 18):
- `limpar_leve`: para embeddings, LLM e citação. Só tira lixo de formatação.
- `limpar_normalizar`: para o BM25. A função da PLN aula 02 (slide 20).
"""
import re
import unicodedata

# --- PII: versão enxuta do detector da IA Gen aula 06 (sem validação aritmética) ---
PADROES_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "TELEFONE": re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?[\s.-]?9?\d{4}[\s.-]\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
}

# Mapa de contrações do PT-BR informal (PLN aula 02, slide 15)
CONTRACTIONS = {
    "vc": "você", "tb": "também", "tbm": "também", "pq": "por que", "obg": "obrigado",
    "blz": "beleza", "td": "tudo", "msm": "mesmo", "vlw": "valeu", "n": "não",
    "naum": "não", "mt": "muito", "pra": "para",
}


def mascarar_pii(texto):
    """Troca cada PII por [TIPO]; devolve o texto e a lista de tipos encontrados."""
    achados = []
    for tipo, padrao in PADROES_PII.items():
        texto, n = padrao.subn(f"[{tipo}]", texto)
        achados += [tipo] * n
    return texto, achados


def limpar_leve(texto):
    """Para embeddings/LLM: só formatação. Caixa, acentos e pontuação ficam."""
    texto = texto.replace("\\r\\n", " ").replace("\\n", " ")  # \r\n escrito como texto
    texto = re.sub(r"\s+", " ", texto)                        # \r\n reais, tabs, espaços repetidos
    return texto.strip()


def limpar_normalizar(texto, remove_acentos=False):
    """Pipeline modular para texto em PT-BR (PLN aula 02, slide 20)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()                                            # 1. minúsculas
    texto = re.sub(r"<[^>]+>", " ", texto)                           # 2. HTML
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)                  # 3. URLs
    texto = re.sub(r"\S+@\S+", " ", texto)                           # 4. e-mails
    texto = " ".join(CONTRACTIONS.get(p, p) for p in texto.split())  # 5. contrações
    texto = re.sub(r"[^a-záàâãäéèêëíìîïóòôõöúùûüç\s.,!?]", " ", texto)  # 6. só letras e pontuação
    if remove_acentos:                                               # 7. acentos (opcional)
        nfkd = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip()                        # 8. espaços
