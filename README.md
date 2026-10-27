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
(`StateGraph`), como nas aulas. O LLM padrão é o Gemini 2.5 Flash, com fallback para a
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
pytest                            # 77 testes (LLM e embeddings falsos: sem rede)
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
│   └── rag.py        #   o grafo LangGraph e o log de auditoria
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
  o BM25. É uma checagem de sanidade; a avaliação final do RAG terá um gold set com
  julgamento humano.
- **Recusa:** a maior similaridade separou bem as perguntas do tema (0,73–0,997) das
  de fora (0,48–0,54). O limiar `LIMIAR_EVIDENCIA = 0,55` é provisório.

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
| Recusa com duas barreiras: limiar + juiz LLM, mínimo de 2 evidências | o limiar não separa pergunta do domínio sem resposta ("entrega por drone") |
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
