"""Criação do LLM de chat, com fallback entre provedores (LangChain aulas 03 e 05).

A chave fica no `.env` da raiz do projeto (fora do git; modelo em `.env.example`).
`LLM_PROVIDER` aceita 'google', 'openai' ou 'auto' (padrão): tenta o Gemini e cai
para a OpenAI se ele não responder — chave ausente, cota estourada ou modelo fora.
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter

from voc_rag.config import RAIZ

MODELO_GEMINI_PADRAO = "models/gemini-3.8-flash"   # o 2.5 não é mais liberado para contas novas
MODELO_OPENAI_PADRAO = "gpt-4o-mini"
REQUISICOES_POR_MINUTO = 8   # abaixo do limite do plano gratuito do Gemini (~10/min)


def _limitador():
    """Espaça as chamadas: o Multi-Query faz várias por pergunta e estouraria a cota (429)."""
    por_minuto = float(os.environ.get("LLM_RPM") or REQUISICOES_POR_MINUTO)
    return InMemoryRateLimiter(requests_per_second=por_minuto / 60, check_every_n_seconds=0.1)


def _mascarar_segredos(texto):
    """Tira credenciais de mensagens de erro antes de imprimir."""
    return re.sub(r"(key=|AIza|sk-)[A-Za-z0-9_\-]{6,}", r"\1<REDACTED>", str(texto))


def _validar(llm, rotulo):
    """Uma chamada mínima: a cota estourada aparece aqui, não no meio do pipeline."""
    try:
        llm.invoke("ping")
        print(f"LLM OK: {rotulo}")
        return llm
    except Exception as erro:
        print(f"{rotulo} indisponível -> {type(erro).__name__}: {_mascarar_segredos(erro)}")
        return None


def _criar_google(validar):
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY não encontrada no .env.")
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    modelo = os.environ.get("GEMINI_CHAT_MODEL") or MODELO_GEMINI_PADRAO
    llm = ChatGoogleGenerativeAI(model=modelo, temperature=0, rate_limiter=_limitador())
    return _validar(llm, f"Google / {modelo}") if validar else llm


def _criar_openai(validar):
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY não encontrada no .env.")
        return None
    from langchain_openai import ChatOpenAI

    modelo = os.environ.get("OPENAI_CHAT_MODEL") or MODELO_OPENAI_PADRAO
    llm = ChatOpenAI(model=modelo, temperature=0, rate_limiter=_limitador())
    return _validar(llm, f"OpenAI / {modelo}") if validar else llm


def criar_llm(provedor=None, validar=True):
    """Devolve o chat model escolhido (temperature=0: o RAG quer respostas reprodutíveis).

    Levanta RuntimeError se nenhum provedor estiver disponível.
    """
    load_dotenv(RAIZ / ".env")
    provedor = (provedor or os.environ.get("LLM_PROVIDER") or "auto").lower()
    if provedor == "google":
        llm = _criar_google(validar)
    elif provedor == "openai":
        llm = _criar_openai(validar)
    else:
        llm = _criar_google(validar) or _criar_openai(validar)
    if llm is None:
        raise RuntimeError("Nenhum LLM disponível: crie o .env a partir do .env.example "
                           "com GOOGLE_API_KEY (ou OPENAI_API_KEY).")
    return llm
