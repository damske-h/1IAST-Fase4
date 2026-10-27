import pytest

from voc_rag import llm


def test_criar_llm_sem_chave_explica_o_que_fazer(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    for var in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match=".env.example"):
        llm.criar_llm()


def test_criar_llm_google_sem_validar(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GOOGLE_API_KEY", "chave-de-teste")
    modelo = llm.criar_llm(provedor="google", validar=False)
    assert "gemini-2.5-flash" in modelo.model and modelo.temperature == 0


def test_mascarar_segredos():
    assert "AIzaSyABCDEF123" not in llm._mascarar_segredos("erro key=AIzaSyABCDEF123 inválida")
