"""LLM falso para os testes: devolve respostas prontas no schema pedido, sem rede nem chave."""
from langchain_core.runnables import RunnableLambda


class LLMFalso:
    """Imita `chat_model.with_structured_output(Schema)`.

    `resposta` é um dict (sempre o mesmo) ou uma função prompt_texto -> dict. Para um
    pipeline com vários schemas, passe {"NomeDoSchema": resposta, ...}.
    Guarda os prompts recebidos em `self.prompts` para os testes olharem.
    """

    def __init__(self, resposta):
        self.resposta = resposta
        self.prompts = []

    def with_structured_output(self, schema):
        def responder(prompt_value):
            texto = prompt_value.to_string()
            self.prompts.append(texto)
            resposta = self.resposta
            if isinstance(resposta, dict) and schema.__name__ in resposta:
                resposta = resposta[schema.__name__]
            dados = resposta(texto) if callable(resposta) else resposta
            return schema.model_validate(dados)
        return RunnableLambda(responder)
