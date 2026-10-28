"""Demo do RAG de Voice of Customer da Olist.

    streamlit run app.py

Pergunte em linguagem natural; o RAG (o grafo LangGraph do notebook 09, sistema H5
da avaliação) responde citando as reviews, recusa quando não há evidência e bloqueia
entradas perigosas. A aba de avaliação lê o cache versionado e não chama o LLM.
"""
import streamlit as st

from voc_rag import demo, rag

st.set_page_config(page_title="Voice of Customer Olist", layout="wide")
st.title("Voice of Customer: pergunte às reviews da Olist")

recursos = st.cache_resource(show_spinner="Carregando base, índice e LLM (só na primeira vez)...")(
    demo.carregar_recursos)


def tela_rag(documentos, app):
    """A aba de perguntas: filtros na barra lateral, exemplos, resposta e trilha."""
    opcoes = demo.opcoes_da_base(documentos)

    with st.sidebar:
        st.header("Filtros (opcionais)")
        st.caption("Sem filtros, o roteador extrai da própria pergunta (UF, período, nota, categoria). "
                   "O que for escolhido aqui tem prioridade.")
        estados = st.multiselect("Estados", opcoes["estados"], placeholder="Todos")
        categorias = st.multiselect("Categorias", opcoes["categorias"], placeholder="Todas")
        notas = st.slider("Nota", 1, 5, (1, 5))
        periodo = st.select_slider("Período", opcoes["meses"], (opcoes["meses"][0], opcoes["meses"][-1]))
        atraso = st.radio("Entrega", list(demo.ATRASO), horizontal=True)
        mostrar_trilha = st.checkbox("Mostrar o caminho no grafo", value=True)

    st.write("Exemplos:")
    colunas = st.columns(3)
    for i, exemplo in enumerate(demo.PERGUNTAS_EXEMPLO):
        if colunas[i % 3].button(exemplo, key=f"exemplo_{i}", width="stretch"):
            st.session_state["pergunta"] = exemplo

    with st.form("pergunta_form"):
        pergunta = st.text_area("Pergunta", key="pergunta", height=80,
                                placeholder="Ex.: o que os clientes reclamam sobre a embalagem?")
        enviar = st.form_submit_button("Perguntar", type="primary")

    if enviar and pergunta.strip():
        filtros = demo.montar_filtros(opcoes, estados, categorias, notas, periodo, atraso)
        with st.spinner("Consultando: guardrail, roteador, Multi-Query, evidência e geração..."):
            estado = rag.responder(app, pergunta, filtros)
        r = estado["resposta"]

        mostrar = {"respondida": st.success, "recusada": st.warning, "bloqueada": st.error}[r.status]
        mostrar(f"**{r.status.upper()}** · {r.resumo}")
        for ponto in r.pontos:
            st.markdown(f"- {ponto.afirmacao} " + " ".join(f"`{rid[:8]}`" for rid in ponto.review_ids))
        if r.evidencias:
            st.subheader(f"Evidências citadas ({len(r.evidencias)})")
            st.dataframe(demo.tabela_evidencias(r), hide_index=True, width="stretch")
        if r.status != "respondida":
            st.caption(f"Motivo: {r.motivo}")

        if mostrar_trilha:
            with st.expander("Como o RAG chegou aqui", expanded=r.status != "respondida"):
                st.dataframe(demo.passos(estado["trilha"]), hide_index=True, width="stretch")
                st.write("Filtros aplicados:", r.filtros or "nenhum")
                if r.consultas:
                    st.write("Consultas do Multi-Query:", r.consultas)
                if r.avisos:
                    st.write("Avisos:", r.avisos)
                st.caption("Cada pergunta vira uma linha em artifacts/logs/auditoria.jsonl, só com o hash do texto.")


aba_rag, aba_avaliacao = st.tabs(["Perguntar ao RAG", "Resultados da avaliação"])

with aba_rag:
    try:
        documentos, app = recursos()
    except Exception as erro:                       # sem base, índice ou chave: explica, e a outra aba segue
        st.error(f"Não foi possível carregar o RAG ({type(erro).__name__}). Rode `python -m voc_rag` "
                 "para gerar a base e o índice e crie o `.env` com a GOOGLE_API_KEY.")
    else:
        tela_rag(documentos, app)

with aba_avaliacao:
    st.caption("Lido de avaliacao/respostas/ (versionado no git): nenhuma chamada ao LLM. "
               "Detalhes no notebook 10_avaliacao.ipynb.")
    try:
        tabelas = demo.resultados_avaliacao()
    except FileNotFoundError:
        st.info("O cache da avaliação não está nesta cópia do repositório.")
    else:
        st.subheader("Teste trancado (20 perguntas, rodado uma vez): baseline B0 x sistema final H5")
        st.dataframe(tabelas["teste"], width="stretch")
        st.subheader("Dev (40 perguntas): baseline e hipóteses H1 a H5")
        st.dataframe(tabelas["dev"], width="stretch")
        st.subheader("Matriz de recusa no dev")
        st.dataframe(tabelas["dev_recusa"], width="stretch")
