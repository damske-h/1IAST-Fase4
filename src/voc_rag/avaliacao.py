"""Checagem rápida da recuperação: precisão@k com julgamento por padrões de texto.

Inspirada na avaliação da IA Gen aula 02 (recall@k e latência). Aqui não há um
gabarito com "o documento certo": numa pergunta como "o pedido não chegou",
centenas de reviews são relevantes. Então julgamos cada review do top-k com uma
expressão regular sobre o `texto_lexical` (o que conta como "fala do tema").

Limite assumido: o juiz é lexical, então tende a favorecer o BM25. Por isso as
perguntas usam palavras diferentes das do padrão, e o notebook manda ler o top-k
de olho. O gold set do protocolo de avaliação (com julgamento humano) vem depois.
"""
import re

import pandas as pd

from voc_rag.config import LIMIAR_EVIDENCIA

# (pergunta escrita como um analista, padrão que marca uma review como relevante)
PERGUNTAS_CHECAGEM = [
    ("o pedido nunca foi entregue",
     r"nao (foi |ser )?(entreg|cheg|receb)|ainda nao|nunca (cheg|receb)|nao recebi"),
    ("produto chegou com defeito ou danificado",
     r"defeit|quebr|danific|estragad|avari|amassad|rachad|nao funcion"),
    ("entrega mais rápida do que o esperado",
     r"rapid|antes do prazo|adiantad|chegou cedo|super rapid|ligeir"),
    ("recebi um produto diferente do anunciado",
     r"diferente|errad|outro produto|outra cor|nao (e|era) o (mesmo|que)|trocad"),
    ("problema com a nota fiscal da compra",
     r"nota fiscal|cupom fiscal|\bnf\b|\bnfe\b"),
    ("vendedor não responde as mensagens",
     r"nao respond|sem resposta|nenhuma resposta|nao (tive|obtive) retorno|sem retorno|contato"),
    ("comprei mais de um item e só veio uma parte",
     r"falt|incomplet|so (veio|chegou|recebi)|apenas (um|uma|1)|veio (um|uma) so|metade"),
    ("quero meu dinheiro de volta",
     r"estorn|reembols|devolu|dinheiro de volta|ressarc|cancel"),
    ("embalagem chegou toda amassada",
     r"embala|caixa|amassad|pacote"),
    ("material de ótima qualidade, recomendo",
     r"qualidade|otim|excelent|perfeit|recomendo"),
]

PERGUNTAS_FORA_DE_ESCOPO = [
    "qual é a capital da França?",
    "me conte uma piada sobre programadores",
    "como faço um bolo de cenoura?",
    "quem ganhou a copa do mundo de 2002?",
]

# Do domínio (e-commerce, entrega, produto), mas sem resposta esperada nas reviews da
# Olist de 2016-2018: o caso difícil da recusa, em que a similaridade engana.
PERGUNTAS_SEM_EVIDENCIA = [
    "os clientes elogiam a entrega feita por drone?",
    "o que acham do programa de cashback em criptomoeda?",
    "reclamações sobre o atendimento na loja física da Olist",
    "os clientes gostaram da experiência de compra no metaverso?",
]


def precisao_em_k(resultado, padrao):
    """Fração do top-k cujo texto casa com o padrão de relevância."""
    if len(resultado) == 0:
        return 0.0
    from voc_rag.texto import limpar_normalizar

    regex = re.compile(padrao)
    relevantes = [bool(regex.search(limpar_normalizar(t, remove_acentos=True))) for t in resultado["texto"]]
    return sum(relevantes) / len(relevantes)


def checar_recuperacao(buscador, perguntas=PERGUNTAS_CHECAGEM, modos=("vetorial", "bm25", "hibrida"), k=10):
    """Uma linha por (pergunta, modo): precisão@k, similaridade do top-1 e latência."""
    linhas = []
    for pergunta, padrao in perguntas:
        for modo in modos:
            resultado = buscador.buscar(pergunta, k=k, modo=modo)
            linhas.append({
                "pergunta": pergunta, "modo": modo,
                f"precisao@{k}": precisao_em_k(resultado, padrao),
                "similaridade_top1": float(resultado["similaridade"].max()) if len(resultado) else 0.0,
                "latencia_ms": round(resultado.attrs["latencia_ms"], 1),
            })
    return pd.DataFrame(linhas)


def checar_recusa(buscador, dentro=PERGUNTAS_CHECAGEM, fora=PERGUNTAS_FORA_DE_ESCOPO,
                  limiar=LIMIAR_EVIDENCIA):
    """Separa perguntas do tema e fora dele pela maior similaridade do top-10 vetorial."""
    linhas = []
    for grupo, perguntas in (("dentro", [p for p, _ in dentro]), ("fora", list(fora))):
        for pergunta in perguntas:
            similaridade = float(buscador.buscar(pergunta, k=10, modo="vetorial")["similaridade"].max())
            linhas.append({"grupo": grupo, "pergunta": pergunta, "similaridade_top1": round(similaridade, 3),
                           "tem_evidencia": similaridade >= limiar})
    return pd.DataFrame(linhas)
