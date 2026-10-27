"""O RAG inteiro como um grafo de estados com LangGraph (LangChain aula 05).

    guardrail ─bloqueada──────────────────────────────────────────────┐
        │                                                             │
      rotear ─fora_de_escopo──────────────────────┐                   │
        │                                         ▼                   ▼
    recuperar ─> checar ─sem evidência─────────> recusar ─────────> auditar ─> FIM
                    │                                                 ▲
                    └─────────────> gerar ────────────────────────────┘

Cada nó devolve só o que mudou; a `trilha` e os `avisos` se acumulam pelo
reducer (operator.add), como no `EstadoAtendimento` da aula 05. O nó `auditar`
grava uma linha JSON por pergunta, com hash e decisões, nunca o texto (IA Gen aula 06).

Uso pela linha de comando:
    python -m voc_rag.rag "Quais as reclamações sobre atraso na entrega?"
"""
import json
import operator
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from voc_rag import evidencia, geracao, guardrails, multiquery, roteador
from voc_rag.busca import posicoes_permitidas
from voc_rag.config import ARQUIVO_AUDITORIA, K_CONTEXTO, LIMIAR_EVIDENCIA, MIN_EVIDENCIAS
from voc_rag.schemas import Evidencia, RespostaRAG


class EstadoRAG(TypedDict, total=False):
    """O contrato de dados do grafo: entrada, decisões intermediárias e saída."""

    texto: str                         # a pergunta como o analista escreveu
    filtros_explicitos: dict | None
    inicio: float
    avaliacao: Any                     # guardrails.AvaliacaoEntrada
    roteamento: Any                    # schemas.Roteamento
    consultas: list[str]
    recuperados: Any                   # DataFrame do Multi-Query
    decisao: Any                       # evidencia.Decisao
    resposta: RespostaRAG
    avisos: Annotated[list[str], operator.add]
    trilha: Annotated[list[str], operator.add]


def _evidencias(recuperados, ids):
    """As linhas citadas, na ordem dos ids, no formato do contrato."""
    por_id = recuperados.set_index("review_id").astype(object)
    por_id = por_id.where(por_id.notna(), None)              # NaN do pandas -> None do Pydantic
    return [Evidencia(review_id=rid, texto=por_id.at[rid, "texto"], nota=int(por_id.at[rid, "nota"]),
                      categoria=por_id.at[rid, "categoria"], estado=por_id.at[rid, "estado"],
                      ano_mes=por_id.at[rid, "ano_mes"]) for rid in ids]


def criar_rag(documentos, buscador, llm, politica=guardrails.POLITICA_PADRAO,
              limiar=LIMIAR_EVIDENCIA, min_evidencias=MIN_EVIDENCIAS, k=K_CONTEXTO,
              caminho_log=ARQUIVO_AUDITORIA):
    """Monta as chains de cada etapa e compila o grafo. `caminho_log=None` desliga o log."""
    cadeia_roteador = roteador.criar_roteador(llm, documentos)
    gerador_variacoes = multiquery.criar_gerador_variacoes(llm)
    juiz = evidencia.criar_juiz(llm)
    gerador = geracao.criar_gerador(llm)

    def guardrail(estado):
        av = guardrails.avaliar_entrada(estado["texto"], estado.get("filtros_explicitos"), politica)
        saida = {"avaliacao": av, "trilha": [f"guardrail -> {'ok' if av.permitida else 'bloqueada'}"]}
        if not av.permitida:
            saida["resposta"] = RespostaRAG(status="bloqueada", resumo=guardrails.MENSAGEM_BLOQUEIO,
                                            motivo="; ".join(av.motivos))
        return saida

    def rotear(estado):
        rot, avisos = roteador.rotear(cadeia_roteador, estado["avaliacao"].pergunta)
        return {"roteamento": rot, "avisos": avisos,
                "trilha": [f"rotear -> {rot.intencao} {rot.filtros.como_kwargs()}"]}

    def recuperar(estado):
        rot = estado["roteamento"]
        consultas, avisos = multiquery.gerar_consultas(gerador_variacoes, rot.consulta)
        filtros = rot.filtros.como_kwargs()
        permitidos = posicoes_permitidas(documentos, **filtros) if filtros else None
        recuperados = multiquery.buscar_multi_query(buscador, consultas, k=k, permitidos=permitidos)
        return {"consultas": consultas, "recuperados": recuperados, "avisos": avisos,
                "trilha": [f"recuperar -> {len(consultas)} consultas, {len(recuperados)} reviews"
                           + (f" ({len(permitidos)} elegíveis pelos filtros)" if filtros else "")]}

    def checar(estado):
        d = evidencia.checar_evidencia(estado["recuperados"], estado["avaliacao"].pergunta.texto,
                                       juiz=juiz, limiar=limiar, min_evidencias=min_evidencias)
        return {"decisao": d, "avisos": d.avisos,
                "trilha": [f"checar -> {'com' if d.tem_evidencia else 'sem'} evidência "
                           f"({len(d.review_ids)} aprovadas)"]}

    def gerar(estado):
        d, recuperados = estado["decisao"], estado["recuperados"]
        aprovadas = recuperados[recuperados["review_id"].isin(d.review_ids)]
        resumo, pontos, avisos = geracao.gerar_resposta(gerador, estado["avaliacao"].pergunta.texto,
                                                        aprovadas)
        if not pontos:
            resposta = RespostaRAG(status="recusada", resumo=evidencia.MENSAGEM_RECUSA,
                                   motivo="nenhuma afirmação com citação válida")
        else:
            citados = list(dict.fromkeys(r for p in pontos for r in p.review_ids))
            resposta = RespostaRAG(status="respondida", resumo=resumo, pontos=pontos,
                                   evidencias=_evidencias(recuperados, citados), motivo=d.motivo)
        return {"resposta": resposta, "avisos": avisos,
                "trilha": [f"gerar -> {len(pontos)} pontos citados"]}

    def recusar(estado):
        rot = estado["roteamento"]
        if rot.intencao == "fora_de_escopo":
            motivo = "pergunta fora do escopo das reviews"
        else:
            motivo = estado["decisao"].motivo
        return {"resposta": RespostaRAG(status="recusada", resumo=evidencia.MENSAGEM_RECUSA,
                                        motivo=motivo),
                "trilha": [f"recusar -> {motivo[:60]}"]}

    def auditar(estado):
        resposta = estado["resposta"]
        rot = estado.get("roteamento")
        resposta = resposta.model_copy(update={
            "consultas": estado.get("consultas", []),
            "filtros": rot.filtros.como_kwargs() if rot else {},
            "avisos": estado.get("avisos", []),
        })
        latencia_ms = round((time.perf_counter() - estado["inicio"]) * 1000)
        if caminho_log:
            registro = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hash_pergunta": estado["avaliacao"].hash_entrada, "politica": politica.versao,
                "status": resposta.status, "motivo": resposta.motivo,
                "review_ids": [e.review_id for e in resposta.evidencias],
                "filtros": resposta.filtros, "n_consultas": len(resposta.consultas),
                "avisos": resposta.avisos, "trilha": estado["trilha"], "latencia_ms": latencia_ms,
            }
            Path(caminho_log).parent.mkdir(parents=True, exist_ok=True)
            with open(caminho_log, "a", encoding="utf-8") as log:
                log.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        return {"resposta": resposta, "trilha": [f"auditar -> {resposta.status} em {latencia_ms} ms"]}

    grafo = StateGraph(EstadoRAG)
    for nome, no in [("guardrail", guardrail), ("rotear", rotear), ("recuperar", recuperar),
                     ("checar", checar), ("gerar", gerar), ("recusar", recusar), ("auditar", auditar)]:
        grafo.add_node(nome, no)

    grafo.add_edge(START, "guardrail")
    grafo.add_conditional_edges("guardrail", lambda e: "rotear" if e["avaliacao"].permitida else "auditar",
                                ["rotear", "auditar"])
    grafo.add_conditional_edges(
        "rotear", lambda e: "recusar" if e["roteamento"].intencao == "fora_de_escopo" else "recuperar",
        ["recuperar", "recusar"])
    grafo.add_edge("recuperar", "checar")
    grafo.add_conditional_edges("checar", lambda e: "gerar" if e["decisao"].tem_evidencia else "recusar",
                                ["gerar", "recusar"])
    grafo.add_edge("gerar", "auditar")
    grafo.add_edge("recusar", "auditar")
    grafo.add_edge("auditar", END)
    return grafo.compile()


def responder(app, texto, filtros=None):
    """Roda o grafo para uma pergunta. Devolve o estado final (resposta em estado["resposta"])."""
    return app.invoke({"texto": texto, "filtros_explicitos": filtros, "inicio": time.perf_counter(),
                       "avisos": [], "trilha": []})


def imprimir(resposta):
    """Mostra a resposta como o analista leria: resumo, pontos com [review_id] e evidências."""
    print(f"[{resposta.status.upper()}] {resposta.resumo}")
    for ponto in resposta.pontos:
        print(f"  - {ponto.afirmacao} [{', '.join(ponto.review_ids)}]")
    if resposta.evidencias:
        print("Evidências:")
        for e in resposta.evidencias:
            print(f"  {e.review_id} (nota {e.nota}, {e.categoria}, {e.estado}, {e.ano_mes}): {e.texto}")
    if resposta.status != "respondida":
        print(f"Motivo: {resposta.motivo}")


if __name__ == "__main__":
    import sys

    from voc_rag import busca, indice, llm
    from voc_rag.preparacao import carregar_base

    documentos = carregar_base()
    indice_vetorial, _ = indice.carregar_indice()
    buscador = busca.Buscador(documentos, indice_vetorial, indice.carregar_modelo())
    app = criar_rag(documentos, buscador, llm.criar_llm())
    pergunta = " ".join(sys.argv[1:]) or "Quais as principais reclamações sobre atraso na entrega?"
    imprimir(responder(app, pergunta)["resposta"])
