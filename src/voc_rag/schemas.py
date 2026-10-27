"""Contratos de dados do RAG com Pydantic (LangChain aulas 02 e 04).

O Pydantic garante a **estrutura**: tipos, faixas e campos permitidos. Ele não
detecta intenção maliciosa (isso é o `guardrails`), mas impede que uma entrada
malformada ou um campo inventado chegue ao Buscador ou ao LLM.

Os mesmos modelos servem de schema para o `with_structured_output` do LangChain
nas próximas etapas: o roteador devolve `FiltrosBusca`, e o gerador, a resposta.
"""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CARACTERES_PERGUNTA = 500

UFS = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA",
       "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}
_ANO_MES = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class FiltrosBusca(BaseModel):
    """Filtros por metadado: os mesmos argumentos de `busca.posicoes_permitidas`."""

    model_config = ConfigDict(extra="forbid")      # campo desconhecido = erro, não é ignorado

    nota_min: int | None = Field(None, ge=1, le=5, description="Nota mínima (1 a 5).")
    nota_max: int | None = Field(None, ge=1, le=5, description="Nota máxima (1 a 5).")
    categorias: list[str] | None = Field(
        None, description="Categorias da Olist em português, ex.: 'cama_mesa_banho'.")
    estados: list[str] | None = Field(None, description="Siglas de UF, ex.: ['SP', 'RJ'].")
    periodo: list[str] | None = Field(
        None, description="Intervalo de ano_mes inclusivo [início, fim], ex.: ['2018-01', '2018-06'].")
    entregue_com_atraso: bool | None = Field(
        None, description="True = só pedidos entregues com atraso; False = só no prazo.")

    @field_validator("estados")
    @classmethod
    def validar_ufs(cls, estados):
        if estados is None:
            return None
        estados = [e.strip().upper() for e in estados]
        invalidas = sorted(set(estados) - UFS)
        if invalidas:
            raise ValueError(f"UF inválida: {invalidas}")
        return estados

    @field_validator("categorias")
    @classmethod
    def validar_categorias(cls, categorias):
        if categorias is None:
            return None
        categorias = [c.strip().lower() for c in categorias]
        if not all(re.fullmatch(r"[a-z0-9_]+", c) for c in categorias):
            raise ValueError("categoria deve ser um identificador da Olist (minúsculas e _)")
        return categorias

    @field_validator("periodo")
    @classmethod
    def validar_periodo(cls, periodo):
        # lista e não tupla: o JSON Schema de lista é aceito por todos os provedores de LLM
        if periodo is not None and (len(periodo) != 2 or not all(_ANO_MES.match(p) for p in periodo)):
            raise ValueError("periodo deve ser ['AAAA-MM', 'AAAA-MM']")
        return periodo

    @model_validator(mode="after")
    def validar_intervalos(self):
        if self.nota_min is not None and self.nota_max is not None and self.nota_min > self.nota_max:
            raise ValueError("nota_min maior que nota_max")
        if self.periodo is not None and self.periodo[0] > self.periodo[1]:
            raise ValueError("periodo com início depois do fim")
        return self

    def como_kwargs(self):
        """Só os filtros preenchidos, prontos para `posicoes_permitidas(documentos, **kwargs)`."""
        kwargs = self.model_dump(exclude_none=True)
        if "periodo" in kwargs:
            kwargs["periodo"] = tuple(kwargs["periodo"])
        return kwargs


class Pergunta(BaseModel):
    """A pergunta do analista, já validada. Nada chega ao RAG sem passar por aqui."""

    model_config = ConfigDict(extra="forbid")

    texto: str = Field(min_length=3, max_length=MAX_CARACTERES_PERGUNTA)
    filtros: FiltrosBusca = Field(default_factory=FiltrosBusca)

    @field_validator("texto", mode="before")
    @classmethod
    def normalizar_texto(cls, texto):
        if not isinstance(texto, str):
            raise ValueError("a pergunta deve ser texto")
        texto = _CONTROLE.sub("", texto)            # caracteres de controle escondem instruções
        return re.sub(r"\s+", " ", texto).strip()


class Roteamento(BaseModel):
    """O que o roteador LLM extrai da pergunta (schema do `with_structured_output`)."""

    intencao: Literal["analise_reviews", "fora_de_escopo"] = Field(
        description="analise_reviews: a pergunta é sobre a experiência dos clientes "
                    "(entrega, produto, atendimento, frete...). fora_de_escopo: qualquer outra coisa.")
    consulta: str = Field(
        description="A pergunta reescrita para a busca, só com o assunto: sem os trechos "
                    "que viraram filtro (UF, período, nota, categoria).")
    filtros: FiltrosBusca = Field(
        default_factory=FiltrosBusca,
        description="Só os filtros que a pergunta pede explicitamente; o resto fica nulo.")


# --- Resposta (etapa 5) -----------------------------------------------------------------
class Ponto(BaseModel):
    """Uma afirmação da resposta e as reviews que a sustentam."""

    afirmacao: str = Field(description="Uma constatação sobre o que os clientes dizem, em português.")
    review_ids: list[str] = Field(
        min_length=1,
        description="review_id das reviews do contexto que sustentam a afirmação. Obrigatório.")


class RespostaGerada(BaseModel):
    """O que o LLM gerador devolve (schema do `with_structured_output`)."""

    resumo: str = Field(description="Resposta direta à pergunta, em 1 a 3 frases, só com o que "
                                    "as reviews do contexto mostram.")
    pontos: list[Ponto] = Field(min_length=1, description="De 1 a 5 afirmações, cada uma citada.")


class Evidencia(BaseModel):
    """Uma review citada, com o que o analista precisa para conferir."""

    review_id: str
    texto: str
    nota: int
    categoria: str
    estado: str | None = None
    ano_mes: str | None = None


class RespostaRAG(BaseModel):
    """O contrato final do RAG: o que a API/notebook devolve e o log de auditoria resume."""

    status: Literal["respondida", "recusada", "bloqueada"]
    resumo: str
    pontos: list[Ponto] = Field(default_factory=list)
    evidencias: list[Evidencia] = Field(default_factory=list)
    motivo: str = ""
    consultas: list[str] = Field(default_factory=list)
    filtros: dict = Field(default_factory=dict)
    avisos: list[str] = Field(default_factory=list)
