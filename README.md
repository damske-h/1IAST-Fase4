# 1IAST-Fase4 — Voice of Customer Intelligence com RAG

Tech Challenge da Fase 4 (FIAP Pós-Tech AI Scientist): um RAG sobre as reviews
da Olist que responde com evidências (`review_id`), é rastreável e recusa
quando não há evidência.

**Estado atual:** a **base de conhecimento** (documentos com metadados, índice
vetorial e busca híbrida com filtros) e o **pipeline RAG** (guardrails, roteador,
Multi-Query, checagem de evidência, geração com citações e recusa, orquestrados
com LangGraph) estão prontos. O próximo passo é a avaliação com gold set.

## Pipeline da base de conhecimento

```
Kaggle (6 CSVs)  ──dados.py──>  data/raw/
                                   │  preparacao.py + texto.py
                                   ▼
             data/processed/base_conhecimento_reviews.csv   (42.138 documentos)
                                   │  indice.py (MiniLM multilíngue)
                                   ▼
             artifacts/indice_vetorial/  embeddings.npy · review_ids.txt · manifesto.json
                                   │  busca.py
                                   ▼
             Buscador: BM25 + vetorial (FAISS) + RRF + filtros por metadado
```

## Pipeline RAG

```
pergunta ─> guardrail ─> rotear ─> recuperar ─> checar ─> gerar ─> auditar ─> RespostaRAG (JSON)
            (PII,        (LLM:     (Multi-Query  (limiar   (LLM: citação      (log sem
             injeção,     filtros   + busca       + juiz    obrigatória e      texto, só
             Pydantic)    Pydantic) híbrida)      LLM)      checada; PII)      hashes)
               │            │                      │
               └ bloqueada  └ fora de escopo ──────┴ sem evidência ─> recusada
```

Tudo com LangChain (`with_structured_output` em schemas Pydantic, LCEL) e LangGraph
(`StateGraph`), como nas aulas. O LLM padrão é o Gemini 3.8 Flash (troque em `GEMINI_CHAT_MODEL`), com fallback para a
OpenAI (`llm.criar_llm`). Pela linha de comando:

```bash
python -m voc_rag.rag "Quais as reclamações sobre atraso na entrega em SP?"
```

## Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt   # instala src/voc_rag em modo editável
python -m voc_rag                 # baixa os dados, gera a base e o índice
pytest                            # 111 testes (LLM e embeddings falsos: sem rede)
copy .env.example .env            # Linux/Mac: cp; preencha GOOGLE_API_KEY
```

- **Token do Kaggle:** em *Kaggle → Settings → API → Create New Token*, salve o
  `kaggle.json` em `C:\Users\<voce>\.kaggle\` (Linux/Mac: `~/.kaggle/`). Sem token,
  baixe pelo botão *Download* do Kaggle e descompacte em `data/raw/`.
- **Modelo:** na primeira execução o `paraphrase-multilingual-MiniLM-L12-v2`
  (~470 MB) é baixado do Hugging Face. Os embeddings levam ~70 s numa CPU comum.
- **Versões exatas:** `requirements-lock.txt` guarda o ambiente em que a base foi
  gerada (Windows, Python 3.14). Para reproduzir igual:
  `pip install -r requirements-lock.txt && pip install -e . --no-deps`.
- **Etapas separadas:** `python -m voc_rag.dados`, `python -m voc_rag.preparacao`,
  `python -m voc_rag.indice`. `python -m voc_rag --recalcular` refaz o índice.
- **Rastreabilidade:** o `manifesto.json` guarda modelo, dimensão, hash SHA-256 da
  base e versões. Se a base ou o modelo mudarem, `carregar_indice` recusa o índice
  antigo. A preparação é determinística: os mesmos CSVs geram o mesmo hash, com
  fim de linha fixo para dar o mesmo arquivo no Windows e no Linux. Outra versão do
  pandas pode formatar o CSV de outro jeito; aí o índice é refeito (~70 s).
- **Chave do LLM:** gere em [Google AI Studio](https://aistudio.google.com/apikey) e
  coloque em `GOOGLE_API_KEY` no `.env` (que está no `.gitignore`). `LLM_RPM` limita as
  chamadas por minuto, para caber no plano gratuito.
- No terminal do Windows, defina `PYTHONIOENCODING=utf-8` para os acentos saírem certos.

Os dados e os artefatos não vão para o git (licença CC BY-NC-SA 4.0 da Olist).

## Demo em Streamlit

```bash
streamlit run app.py
```

- **Perguntar ao RAG:** pergunta em linguagem natural, exemplos prontos (responder,
  recusar, bloquear) e filtros opcionais na barra lateral (UF, categoria, nota, período,
  atraso). Sem filtros, o roteador os extrai da própria pergunta. A resposta traz o
  status, os pontos com os `review_id` citados, a tabela de evidências e, aberto, o
  caminho no grafo (trilha), as consultas do Multi-Query e o motivo da recusa.
- **Resultados da avaliação:** as tabelas do teste trancado e do dev, lidas do cache
  (sem chamar o LLM). Essa aba funciona mesmo sem a base e sem chave.

A tela é só desenho: carregar o RAG, montar filtros e preparar tabelas fica em
`src/voc_rag/demo.py`, testado (inclusive a tela, com o `AppTest` do Streamlit).

## Como reproduzir a avaliação

A avaliação (notebook `10_avaliacao.ipynb`) compara a baseline B0 com as hipóteses H1 a H5
nas 60 perguntas do gold set. Ela foi **rodada uma vez** e tudo ficou versionado em
`avaliacao/`:

| Arquivo | Conteúdo |
|---|---|
| `avaliacao/gold_set.csv` | as 60 perguntas, estrato, partição dev/teste e gabarito |
| `avaliacao/respostas/{particao}_{sistema}.csv` | a resposta de cada sistema a cada pergunta, com chamadas e tokens |
| `avaliacao/respostas/julgamentos.csv` | relevância das reviews recuperadas (pooling) |
| `avaliacao/respostas/{particao}_{sistema}_fidelidade.csv` | veredicto do juiz para cada afirmação |

Há dois modos, escolhidos na `ConfigAvaliacao` da primeira seção do notebook:

- **`modo="cache"` (padrão): custo zero.** O notebook só lê esses CSVs e recalcula as
  métricas e tabelas. Não precisa de chave, nem de baixar os dados ou gerar o índice.
- **`modo="ao_vivo"`: para ver o pipeline rodando.** Por exemplo,
  `ConfigAvaliacao(modo="ao_vivo", max_perguntas=3, sistemas=["B0", "H4"])` roda três
  perguntas com a chave do `.env` de quem executa (alguns centavos). Antes de rodar, o
  notebook mostra o teto de chamadas. O resultado vai para `artifacts/avaliacao/ao_vivo/`,
  fora do git, e não altera o cache.

O **teste trancado** (20 perguntas) só abre com `abrir_teste=True` e foi rodado uma única
vez, com B0 e o sistema final; o notebook apenas lê o resultado gravado. O RAG ao vivo
(notebook 09 e demo) sempre precisa da chave; a avaliação, não.

## Estrutura

```
1IAST-Fase4/
├── data/
│   ├── raw/          # CSVs do Kaggle, intocados (fora do git)
│   ├── interim/      # etapas intermediárias, descartáveis (fora do git)
│   └── processed/    # base de conhecimento pronta (fora do git)
├── artifacts/        # índice vetorial, manifesto e log de auditoria (fora do git)
├── notebooks/        # notebooks didáticos, numerados: explicam e importam src/
├── src/voc_rag/      # código reutilizável, a única fonte de verdade
│   ├── __main__.py   #   python -m voc_rag: o pipeline inteiro
│   ├── config.py     #   caminhos e parâmetros
│   ├── dados.py      #   download e leitura dos CSVs
│   ├── texto.py      #   limpeza de texto e PII
│   ├── preparacao.py #   reviews -> base de conhecimento
│   ├── indice.py     #   embeddings + índice FAISS
│   ├── busca.py      #   BM25, busca híbrida com RRF e filtros por metadado
│   ├── avaliacao.py  #   checagem de recuperação e do sinal de recusa
│   ├── schemas.py    #   contratos Pydantic: pergunta, filtros, roteamento, resposta
│   ├── guardrails.py #   PII, prompt injection e política como código
│   ├── llm.py        #   criar_llm: Gemini com fallback para OpenAI
│   ├── roteador.py   #   pergunta -> intenção, consulta e filtros
│   ├── multiquery.py #   reformulações da consulta + RRF entre as buscas
│   ├── evidencia.py  #   contexto delimitado, juiz LLM e decisão de recusa
│   ├── geracao.py    #   resposta com citações checadas e PII mascarada
│   ├── rag.py        #   o grafo LangGraph e o log de auditoria
│   ├── goldset.py    #   gold set da avaliação: perguntas, estratos e partição
│   ├── sistemas.py   #   baseline B0 e hipóteses H1 a H4, com contagem de chamadas
│   ├── experimento.py #  ConfigAvaliacao, cache, juízes e métricas
│   └── demo.py       #   apoio da demo: recursos, filtros e tabelas da tela
├── avaliacao/        # gold set e respostas da avaliação (versionados: reprodução sem custo)
├── app.py          # demo em Streamlit (streamlit run app.py)
├── tests/            # testes com pytest (modelo falso: rodam sem baixar nada)
├── pyproject.toml    # pacote voc_rag e dependências
├── .env.example      # modelo do .env com a chave do LLM (o .env fica fora do git)
├── requirements.txt  # atalho: instala o pacote em modo editável
└── requirements-lock.txt  # versões exatas do ambiente de referência
```

A regra do projeto: **a lógica fica em `src/voc_rag`, testada; os notebooks
importam essa lógica, mostram o código com `mostrar_codigo()` e explicam cada
decisão**, citando a aula de origem. Cada etapa também roda pela linha de comando
(`python -m voc_rag.<modulo>`), sem uma pasta `scripts/` repetindo código.

## Notebooks

| Notebook | O que ensina | Aulas |
|---|---|---|
| `01_base.ipynb` | seleção, deduplicação, metadados, limpeza leve x normalização, PII → `data/processed/` | PLN 01, 02, 05 · IA Gen 05, 06 |
| `02_indice_vetorial.ipynb` | embeddings, cosseno, FAISS `IndexFlatIP`, pré-filtro por metadado, limites da busca vetorial | IA Gen 01, 02 · PLN 04 · DL 08 |
| `03_busca_hibrida.ipynb` | BM25, RRF, filtros de negócio, precisão@10 e sinal de recusa | PLN 03 · IA Gen 02 |
| `04_guardrail_entrada.ipynb` | schema Pydantic x intenção, PII com validação aritmética, prompt injection, política como código | IA Gen 03, 06 · LangChain 02, 04 |
| `05_roteador.ipynb` | `criar_llm` com fallback, `with_structured_output`, extração de filtros conferida com a base | LangChain 01, 02, 03, 05 |
| `06_multi_query.ipynb` | reformulação da consulta, RRF entre consultas, custo x ganho | LangChain 03 · IA Gen 02 |
| `07_evidencia_recusa.ipynb` | limiar de similaridade, contexto delimitado, juiz LLM, recusa | LangChain 02 · IA Gen 02, 06 |
| `08_geracao_resposta.ipynb` | resposta JSON com citação obrigatória, validação de citações, ataques à saída | LangChain 02, 04 · IA Gen 06 |
| `09_rag_langgraph.ipynb` | `StateGraph`, arestas condicionais, bateria responder/recusar/bloquear, auditoria | LangChain 05 · IA Gen 06 |
| `10_avaliacao.ipynb` | baseline x hipóteses, gold set, acerto@10/MRR, matriz de recusa, fidelidade com juiz, custo, análise de erro, teste trancado | IA Gen 02 · DL 06 |

## A base de conhecimento em números

| | |
|---|---|
| Reviews na Olist | 99.224 |
| Só com nota (sem texto) | 56.537 |
| Texto sem nenhuma letra | 233 |
| `review_id` em mais de um pedido | 306 (unificados) |
| **Documentos na base** | **42.138** (CSV de 20 MB) |
| Maior review | 229 caracteres → sem chunking |
| Reviews com PII mascarada | 2 (telefone) |
| Reviews com menos de 3 palavras | 5.983 (ficam na base, não viram evidência) |

## Resultados da recuperação

Checagem com 10 perguntas de analista (`avaliacao.PERGUNTAS_CHECAGEM`), k = 10, nos
dados reais:

| Busca | Precisão@10 | Latência média |
|---|---|---|
| Vetorial | 0,85 | 14 ms |
| BM25 | 0,91 | 35 ms |
| Híbrida (RRF) | 0,87 | 40 ms |

- **Limite da medida:** o juiz é uma expressão regular por pergunta, o que favorece
  o BM25. É só uma checagem de sanidade; a avaliação do RAG inteiro vem a seguir.
- **Recusa:** a maior similaridade separou bem as perguntas do tema (0,73–0,997) das
  de fora (0,48–0,54). O limiar `LIMIAR_EVIDENCIA = 0,55` é provisório.

## Resultados da avaliação

Protocolo: baseline congelada (B0, o pipeline da IA Gen aula 02), hipóteses que mudam
uma variável por vez, gold set de 60 perguntas em 7 estratos (40 dev, 20 teste
trancado), juízes LLM para relevância e fidelidade. Detalhes e todas as tabelas no
notebook `10_avaliacao.ipynb`; para reproduzir, veja [Como reproduzir a avaliação](#como-reproduzir-a-avaliação).

**Dev (40 perguntas): cada hipótese no estrato que ela previa**

| Sistema | O que muda | Comportamento certo | Agulhas (B) | Paráfrases (C) | Filtros (D) | Falsa recusa / alucinação | Fidelidade | Chamadas |
|---|---|---|---|---|---|---|---|---|
| B0 | baseline: vetorial, recusa no prompt | 0,68 | 0,57 | 0,20 | 0,00 | 12 / 1 | 0,88 | 40 |
| H1 | + busca híbrida | 0,80 | **1,00** | 0,40 | 0,00 | 7 / 1 | 0,91 | 40 |
| H2 | + Multi-Query | 0,83 | **1,00** | **0,60** | 0,00 | 5 / 2 | 0,94 | 80 |
| H3 | + roteador com filtros | **0,90** | 0,86 | 0,40 | **1,00** | 4 / 0 | 0,92 | 102 |
| H4 | + guardrail e checagem de evidência (mín. 2) | 0,73 | 0,14 | 0,00 | **1,00** | 11 / 0 | 0,95 | 113 |
| **H5** | H4 com mínimo de **1** evidência | **0,90** | 0,71 | **0,60** | **1,00** | **4 / 0** | **0,96** | 118 |

- **H1** confirmou a previsão: o BM25 acha as agulhas (termos raros como "menorah",
  "Blu-Ray"). **H2** ganhou nas paráfrases. **H3** acertou 100% dos filtros de metadado.
- **H4** foi a lição da rodada: a checagem de evidência zerou a alucinação, mas exigia
  2 evidências, e uma agulha só tem uma review certa. Recusou 7 perguntas com a
  resposta no top-10. A **análise de erro** virou a hipótese **H5**
  (`MIN_EVIDENCIAS = 1`), que recuperou as agulhas sem alucinação nova.
- H3 e H5 empatam em comportamento; H5 fica com o maior F1 de recusa (0,87), a maior
  fidelidade e é o único que barra 75% dos ataques já no guardrail, antes de gastar LLM.

**Teste trancado (20 perguntas, rodado uma única vez): B0 x H5**

| | B0 | H5 |
|---|---|---|
| Comportamento certo (IC 95% bootstrap) | 0,80 (0,60–0,95) | **0,90** (0,75–1,00) |
| F1 da recusa | 0,78 | **0,88** |
| Alucinação | 0 | 0 |
| Precisão@10 | 0,39 | **0,69** |
| Fidelidade (juiz) | 0,85 | **1,00** |
| Filtros exatos (D) | 0,00 | **0,67** |
| Latência mediana / p95 | 7 s / 17 s | 25 s / 39 s |
| Chamadas ao LLM | 20 | 58 |

Pergunta a pergunta, o H5 vence 2, empata 18 e não perde nenhuma. Com 20 perguntas os
intervalos se sobrepõem: o ganho é consistente, mas a amostra é pequena para chamá-lo
de significativo. **Custo total da avaliação:** cerca de R$ 14 no Gemini 3.8 Flash
(~750 chamadas); ~77% dos tokens de saída são o raciocínio interno do modelo.

## Decisões de projeto

| Decisão | Por quê |
|---|---|
| Só reviews com texto viram documento; título + mensagem num texto só | review sem texto não é evidência; o título carrega sentido |
| Uma linha por `review_id` | a citação precisa ser única; cópias contariam em dobro |
| `texto` (limpeza leve + PII) para embedding, `texto_lexical` (normalizado, sem acento) para BM25 | embeddings preferem texto natural; BM25 precisa casar "nao" e "não" |
| 1 review = 1 documento, sem chunking | reviews têm no máximo 229 caracteres |
| Metadados fora do texto, usados como pré-filtro | filtro restritivo não esvazia o top-k; chunk enriquecido fica como hipótese |
| MiniLM multilíngue + FAISS `IndexFlatIP` | o modelo da IA Gen 02; busca exata é rápida com ~42 mil vetores |
| Grava embeddings + manifesto, não o binário do FAISS | o caro é o embedding; o índice flat remonta em milissegundos |
| BM25 montado em memória | 0,1 s sobre a base inteira; sem arquivo para sair de sincronia |
| RRF com k = 60 sobre os 50 primeiros de cada busca | funde por posição, não por pontuações em escalas diferentes |
| Reviews com menos de 3 palavras não viram evidência | com pergunta curta, a vetorial as puxa com similaridade alta e sem sentido |
| Toda saída de LLM em schema Pydantic (`with_structured_output`) | sem parsing de texto livre; campo inventado vira erro (`extra="forbid"`) |
| Saída do LLM conferida com dados reais | categoria inexistente, período fora da base e `review_id` não recuperado são descartados |
| Recusa com duas barreiras: limiar + juiz LLM, mínimo de 1 evidência | o limiar não separa pergunta do domínio sem resposta ("entrega por drone"); mínimo de 2 recusava agulhas (H4 x H5 na avaliação) |
| Avaliação com baseline, uma variável por vez e teste trancado | ganho com causa conhecida; o teste não é usado para ajustar nada |
| Guardrail antes de qualquer LLM; log só com hash | defesa em profundidade e LGPD (IA Gen 06) |

## Limitações conhecidas

- **Pergunta de uma palavra** ("boleto"): na busca simples, o lado vetorial traz elogios
  genéricos (~0,84). No RAG, o Multi-Query reformula a consulta e o juiz descarta
  evidência genérica (notebooks 06 e 07).
- **Detectores por padrão** (PII, injeção) têm falso positivo e falso negativo; são a
  primeira camada, não a única (notebook 04).
- **Nota x texto:** há reviews em que a nota contradiz o texto ("Não foi entregue o
  meu pedido" com nota 5). `sentimento_nota` vem da nota; um classificador de texto
  (PLN 05) pode marcar essas divergências.
- **PII:** o detector é por regex, sem validação aritmética; pode mascarar um número
  de pedido como telefone (falso positivo aceito).
- **Avaliação:** os rótulos de relevância e a fidelidade vêm de juízes LLM; o protocolo
  pede revisão humana de uma amostra (o notebook 10 sorteia 20 rótulos para isso), que
  ainda não foi feita. Duas paráfrases (C02 e C07) nenhum sistema acerta: a review
  certa não chega ao top-10. O teste tem só 20 perguntas.
- **Latência e custo:** o H5 leva ~25 s por pergunta (4 chamadas ao LLM, espaçadas
  pelo limitador) contra ~7 s da baseline.

## Dicionário de campos da base

| Campo | Uso no RAG |
|---|---|
| `review_id` | **citação** e id do chunk |
| `order_ids` | rastrear até o pedido (mais de um quando a review foi compartilhada) |
| `texto` | entra no embedding e é a evidência mostrada ao usuário |
| `texto_lexical` | entra no BM25 |
| `titulo_original`, `mensagem_original` | auditoria: o texto exatamente como veio da Olist |
| `nota`, `sentimento_nota` | filtro ("só reclamações") |
| `data_review`, `ano_mes` | filtro de período |
| `status_pedido`, `dias_entrega`, `atraso_dias`, `entregue_com_atraso` | filtro e contexto de entrega |
| `categoria`, `categoria_en`, `n_categorias` | filtro por categoria de produto |
| `estado`, `cidade` | filtro por região |
| `n_palavras`, `pii_mascarada` | qualidade da evidência e auditoria de LGPD |
