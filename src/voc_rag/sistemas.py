"""Os sistemas comparados na avaliação: a baseline B0 e as hipóteses H1 a H4.

Como no protocolo de avaliação (e nos notebooks de Deep Learning), cada sistema
muda uma variável em relação ao anterior, para o ganho ter causa conhecida:

| Sistema | Busca | Multi-Query | Filtros (roteador) | Recusa | Chamadas ao LLM |
|---|---|---|---|---|---|
| B0 | vetorial (aula 02) | não | não | só no prompt | 1 |
| H1 | híbrida BM25 + vetorial | não | não | só no prompt | 1 |
| H2 | híbrida | sim | não | só no prompt | 2 |
| H3 | híbrida | sim | sim | prompt + roteador (fora de escopo) | 3 |
| H4 | o RAG completo (notebook 09) | sim | sim | guardrail + limiar + juiz | 4 |
| H5 | H4 com `MIN_EVIDENCIAS=1` | sim | sim | idem, uma review basta | 4 |

H5 nasceu da análise de erro da rodada dev: o H4 recusava agulhas mesmo com a
review certa no top-10, porque exigia 2 evidências e a agulha só tem uma. H5 venceu
no dev e virou o padrão do RAG (`config.MIN_EVIDENCIAS = 1`); o H4 fica fixo em 2.

B0 a H3 usam o gerador da aula 02: responde só com as reviews, cita os ids e
recusa pelo próprio prompt. As citações saem como o LLM escreveu, sem filtro,
para a avaliação medir quantas são inválidas. H4 é o grafo do `rag.py`, que soma o
guardrail de entrada (sem LLM), a checagem de evidência e a validação das citações.

Cada resposta vira um `Resultado`, com chamadas e tokens contados por um
callback do LangChain: é o gasto real, para comparar com a estimativa.
"""
import json
import time
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from voc_rag import multiquery, rag, roteador
from voc_rag.busca import posicoes_permitidas
from voc_rag.config import K_CONTEXTO, MIN_EVIDENCIAS
from voc_rag.evidencia import MENSAGEM_RECUSA, formatar_contexto
from voc_rag.llm import _mascarar_segredos
from voc_rag.schemas import Pergunta, Ponto

SISTEMAS = {
    "B0": "baseline da aula 02: busca vetorial e recusa no prompt",
    "H1": "B0 + busca híbrida (BM25 + vetorial com RRF)",
    "H2": "H1 + Multi-Query",
    "H3": "H2 + roteador com filtros por metadado",
    "H4": "H3 + guardrail e checagem de evidência (o RAG completo)",
    "H5": "H4 com MIN_EVIDENCIAS=1: uma review basta (agulhas)",
}
# Máximo por pergunta; recusas e bloqueios gastam menos (o H4 bloqueado não chama nada).
CHAMADAS_POR_PERGUNTA = {"B0": 1, "H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 4}


# --- Gerador da baseline (aula 02) ------------------------------------------------------
class RespostaSimples(BaseModel):
    """Saída do gerador de B0 a H3: a recusa é decidida pelo próprio LLM."""

    recusou: bool = Field(description="true se as reviews não respondem à pergunta.")
    resumo: str = Field(description="Resposta direta em 1 a 3 frases, ou a frase de recusa.")
    pontos: list[Ponto] = Field(default_factory=list,
                                description="Até 5 afirmações citadas; vazio quando recusar.")


PROMPT_SIMPLES = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um analista de Voice of Customer da Olist. Responda à pergunta usando SOMENTE as "
     "reviews entre <reviews></reviews>. Cada afirmação em `pontos` cita em `review_ids` os ids "
     "das reviews que a sustentam. Se as reviews não respondem à pergunta, marque recusou=true, "
     "deixe `pontos` vazio e escreva no resumo: \"{recusa}\". Responda em português."),
    ("human", "Pergunta: <pergunta>{pergunta}</pergunta>\n\n<reviews>\n{contexto}\n</reviews>"),
])


def criar_gerador_simples(llm):
    """Chain LCEL: {"pergunta", "contexto"} -> RespostaSimples."""
    return PROMPT_SIMPLES.partial(recusa=MENSAGEM_RECUSA) | llm.with_structured_output(RespostaSimples)


def _gerar_simples(gerador, pergunta, recuperados):
    """Gera com o prompt da aula 02. Citações ficam como vieram: a avaliação as confere."""
    if len(recuperados) == 0:
        return {"status": "recusada", "resumo": MENSAGEM_RECUSA}
    bruta = gerador.invoke({"pergunta": pergunta, "contexto": formatar_contexto(recuperados)})
    if bruta.recusou or not bruta.pontos:
        return {"status": "recusada", "resumo": bruta.resumo or MENSAGEM_RECUSA}
    return {"status": "respondida", "resumo": bruta.resumo, "pontos": bruta.pontos,
            "citados": list(dict.fromkeys(r for p in bruta.pontos for r in p.review_ids))}


# --- Os sistemas -------------------------------------------------------------------------
def criar_sistemas(documentos, buscador, llm, nomes=tuple(SISTEMAS), k=K_CONTEXTO):
    """Monta cada sistema pedido como uma função `pergunta -> dict` (campos do `Resultado`)."""
    gerador = criar_gerador_simples(llm)
    gerador_variacoes = multiquery.criar_gerador_variacoes(llm)
    cadeia_roteador = roteador.criar_roteador(llm, documentos)

    def busca_simples(modo):
        def sistema(pergunta):
            recuperados = buscador.buscar(pergunta, k=k, modo=modo)
            return {**_gerar_simples(gerador, pergunta, recuperados),
                    "recuperados": recuperados["review_id"].tolist(), "consultas": [pergunta]}
        return sistema

    def com_multi_query(pergunta, consulta=None, filtros=None):
        consultas, avisos = multiquery.gerar_consultas(gerador_variacoes, consulta or pergunta)
        permitidos = posicoes_permitidas(documentos, **filtros) if filtros else None
        recuperados = multiquery.buscar_multi_query(buscador, consultas, k=k, permitidos=permitidos)
        saida = _gerar_simples(gerador, pergunta, recuperados)
        return {**saida, "recuperados": recuperados["review_id"].tolist(), "consultas": consultas,
                "filtros": filtros or {}, "avisos": avisos}

    def h3(pergunta):
        rot, avisos = roteador.rotear(cadeia_roteador, Pergunta(texto=pergunta))
        if rot.intencao == "fora_de_escopo":
            return {"status": "recusada", "resumo": MENSAGEM_RECUSA, "avisos": avisos}
        saida = com_multi_query(pergunta, rot.consulta, rot.filtros.como_kwargs())
        return {**saida, "avisos": avisos + saida["avisos"]}

    def grafo(min_evidencias):
        app = rag.criar_rag(documentos, buscador, llm, k=k, min_evidencias=min_evidencias, caminho_log=None)

        def sistema(pergunta):
            estado = rag.responder(app, pergunta)
            r = estado["resposta"]
            recuperados = estado.get("recuperados")
            return {"status": r.status, "resumo": r.resumo, "pontos": r.pontos, "motivo": r.motivo,
                    "recuperados": [] if recuperados is None else recuperados["review_id"].tolist(),
                    "citados": [e.review_id for e in r.evidencias], "consultas": r.consultas,
                    "filtros": r.filtros, "avisos": r.avisos}
        return sistema

    construtores = {"B0": lambda: busca_simples("vetorial"), "H1": lambda: busca_simples("hibrida"),
                    "H2": lambda: com_multi_query, "H3": lambda: h3,
                    "H4": lambda: grafo(2), "H5": lambda: grafo(MIN_EVIDENCIAS)}
    return {nome: construtores[nome]() for nome in nomes}


# --- Contagem de chamadas e tokens -----------------------------------------------------------
class ContadorLLM(BaseCallbackHandler):
    """Callback do LangChain: conta chamadas ao chat model e soma os tokens informados."""

    def __init__(self):
        self.chamadas = 0
        self.tokens_entrada = 0
        self.tokens_saida = 0
        self.tokens_raciocinio = 0   # modelos que "pensam" cobram o raciocínio como saída

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.chamadas += 1

    def on_llm_end(self, response, **kwargs):
        for geracoes in response.generations:
            for geracao in geracoes:
                uso = getattr(getattr(geracao, "message", None), "usage_metadata", None) or {}
                self.tokens_entrada += uso.get("input_tokens", 0)
                self.tokens_saida += uso.get("output_tokens", 0)
                self.tokens_raciocinio += (uso.get("output_token_details") or {}).get("reasoning", 0)


class Resultado(BaseModel):
    """Uma pergunta respondida por um sistema: o que a avaliação mede e o cache guarda."""

    sistema: str
    id: str
    status: Literal["respondida", "recusada", "bloqueada", "erro"]
    resumo: str = ""
    motivo: str = ""                                       # H4/H5: por que respondeu ou recusou
    pontos: list[Ponto] = Field(default_factory=list)
    recuperados: list[str] = Field(default_factory=list)   # o top-k que foi ao LLM, em ordem
    citados: list[str] = Field(default_factory=list)       # como saíram da resposta
    consultas: list[str] = Field(default_factory=list)
    filtros: dict = Field(default_factory=dict)
    avisos: list[str] = Field(default_factory=list)
    chamadas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    tokens_raciocinio: int = 0
    latencia_ms: int = 0

    def como_linha(self):
        """Linha de CSV: listas e dicts viram JSON."""
        linha = self.model_dump()
        for campo in ("pontos", "recuperados", "citados", "consultas", "filtros", "avisos"):
            linha[campo] = json.dumps(linha[campo], ensure_ascii=False, default=list)
        return linha


def responder(nome, sistema, item):
    """Roda um sistema numa pergunta do gold set, medindo tempo, chamadas e tokens.

    O `RunnableLambda` passa o callback para todas as chains de dentro (inclusive as
    do grafo do H4). Um erro vira status "erro" em vez de parar a avaliação.
    """
    contador = ContadorLLM()
    inicio = time.perf_counter()
    try:
        saida = RunnableLambda(sistema, name=nome).invoke(item.pergunta, config={"callbacks": [contador]})
    except Exception as erro:
        saida = {"status": "erro", "avisos": [f"{type(erro).__name__}: {_mascarar_segredos(erro)[:120]}"]}
    return Resultado(sistema=nome, id=item.id, **saida, chamadas=contador.chamadas,
                     tokens_entrada=contador.tokens_entrada, tokens_saida=contador.tokens_saida,
                     tokens_raciocinio=contador.tokens_raciocinio,
                     latencia_ms=round((time.perf_counter() - inicio) * 1000))
