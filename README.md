# Voice of Customer Intelligence com RAG

> **Tech Challenge · Fase 4 | Pós-Tech FIAP | AI Scientist**

**Equipe:** Lucas Damasceno da Cunha Lima (RM372040)

Um RAG sobre as reviews de clientes do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
O analista pergunta em linguagem natural ("quais as reclamações de atraso em SP no
primeiro semestre de 2018?") e recebe uma resposta em que **cada afirmação cita as
reviews que a sustentam** (`review_id`). Quando a base não tem evidência, o sistema
**recusa** em vez de inventar; quando a pergunta é perigosa (dado pessoal, prompt
injection), **bloqueia** antes de chamar o LLM. Cada decisão fica num log de auditoria.

| Resultado no teste trancado (20 perguntas) | Baseline | RAG final |
|---|---|---|
| Comportamento certo (responder, recusar ou bloquear) | 0,80 | **0,90** |
| Alucinação (respondeu sem evidência) | 0 | 0 |
| Precisão@10 da recuperação | 0,39 | **0,69** |
| Fidelidade das afirmações às reviews citadas | 0,85 | **1,00** |

## Sumário

- [Problema de negócio](#problema-de-negócio)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Notebooks: o projeto passo a passo](#notebooks-o-projeto-passo-a-passo)
- [Exemplos de perguntas e respostas](#exemplos-de-perguntas-e-respostas)
- [A base de conhecimento em números](#a-base-de-conhecimento-em-números)
- [Avaliação](#avaliação)
- [Decisões de projeto](#decisões-de-projeto)
- [Limitações conhecidas](#limitações-conhecidas)
- [Dicionário de campos da base](#dicionário-de-campos-da-base)

**Notas técnicas em `docs/`:** [pipeline RAG](docs/pipeline_rag.md) (nós, contratos e
parâmetros), [segurança e LGPD](docs/seguranca.md) (camadas de defesa e log de
auditoria) e [protocolo de avaliação](docs/avaliacao.md) (gold set, sistemas, métricas e
pendências).

## Problema de negócio

Um e-commerce recebe dezenas de milhares de reviews. A nota de 1 a 5 diz **se** o
cliente está satisfeito; o **porquê** está no texto, que ninguém consegue ler em escala.
Este projeto transforma esse texto numa base consultável, para perguntas como:

- Quais são os principais problemas relatados pelos clientes?
- O que dizem os clientes cujo pedido atrasou? E os de uma região, categoria ou período?
- Algum cliente relatou um problema específico (ex.: recebeu o modelo errado)?
- Quais reviews sustentam essa conclusão?

Requisitos que guiam o projeto: toda resposta cita as reviews usadas; pergunta sem
suporte na base recebe uma recusa; filtros por metadado (nota, categoria, estado,
período, atraso); rastreabilidade de ponta a ponta; e nenhum dado pessoal no índice, na
resposta ou no log.

## Arquitetura

**Base de conhecimento** (offline, sem LLM):

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

**Pipeline RAG** (um grafo de estados do LangGraph):

```
pergunta ─> guardrail ─> rotear ─> recuperar ─> checar ─> gerar ─> auditar ─> RespostaRAG (JSON)
            (PII,        (LLM:     (Multi-Query  (limiar   (LLM: citação      (log sem
             injeção,     filtros   + busca       + juiz    obrigatória e      texto, só
             Pydantic)    Pydantic) híbrida)      LLM)      checada; PII)      hashes)
               │            │                      │
               └ bloqueada  └ fora de escopo ──────┴ sem evidência ─> recusada
```

1. **Guardrail** (sem LLM): bloqueia PII confirmada por validação aritmética, sinais de
   prompt injection e tópicos proibidos, segundo uma política versionada.
2. **Roteador** (LLM): extrai a intenção, a consulta e os filtros num schema Pydantic;
   o que o LLM devolve é conferido com a base (categoria inventada é descartada).
3. **Recuperação**: o LLM reformula a consulta (Multi-Query), cada versão passa pela
   busca híbrida BM25 + vetorial, e o RRF funde tudo num ranking só.
4. **Checagem de evidência**: um limiar de similaridade e um juiz LLM decidem se as
   reviews respondem de fato; sem evidência, o RAG recusa.
5. **Geração** (LLM): a resposta sai num schema em que cada afirmação exige `review_id`;
   o código remove citações fora das evidências e mascara PII.
6. **Auditoria**: uma linha JSON por pergunta, com hash, decisões e citações, nunca o
   texto.

Toda saída de LLM passa por `with_structured_output` com schemas Pydantic. O LLM padrão
é o Gemini 3.8 Flash (troque em `GEMINI_CHAT_MODEL`), com fallback para a OpenAI.

## Tecnologias

| Componente | Tecnologia | Uso |
|---|---|---|
| Linguagem | Python 3.10+ | todo o código em `src/voc_rag` |
| Dados | `pandas`, `numpy`, `kagglehub` | download, junção e limpeza dos CSVs |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | vetores de 384 dimensões, locais e gratuitos |
| Busca vetorial | `faiss-cpu` (`IndexFlatIP`) | busca exata por cosseno, com pré-filtro |
| Busca lexical | `rank-bm25` | termos exatos e raros |
| LLM | Gemini via `langchain-google-genai` (fallback `langchain-openai`) | roteador, Multi-Query, juiz e gerador |
| Orquestração | `langchain-core` (LCEL), `langgraph` (`StateGraph`) | chains e o grafo do RAG |
| Contratos | `pydantic` | entrada, filtros, saídas do LLM e resposta final |
| Demo | `streamlit` | perguntas ao RAG e resultados da avaliação |
| Testes | `pytest` | 111 testes com LLM e embeddings falsos (sem rede) |

## Estrutura do repositório

```
1IAST-Fase4/
├── data/
│   ├── raw/          # CSVs do Kaggle, intocados (fora do git)
│   ├── interim/      # etapas intermediárias, descartáveis (fora do git)
│   └── processed/    # base de conhecimento pronta (fora do git)
├── artifacts/        # índice vetorial, manifesto e log de auditoria (fora do git)
├── notebooks/        # 01 a 10: o projeto passo a passo; importam src/
├── src/voc_rag/      # código reutilizável, a única fonte de verdade
│   ├── __main__.py   #   python -m voc_rag: a base de conhecimento inteira
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
│   ├── sistemas.py   #   baseline B0 e hipóteses H1 a H5, com contagem de chamadas
│   ├── experimento.py #  ConfigAvaliacao, cache, juízes e métricas
│   └── demo.py       #   apoio da demo: recursos, filtros e tabelas da tela
├── avaliacao/        # gold set e respostas da avaliação (versionados: reprodução sem custo)
├── docs/             # notas técnicas: pipeline, segurança e protocolo de avaliação
├── app.py            # demo em Streamlit (streamlit run app.py)
├── tests/            # testes com pytest (modelo falso: rodam sem baixar nada)
├── pyproject.toml    # pacote voc_rag e dependências
├── .env.example      # modelo do .env com a chave do LLM (o .env fica fora do git)
├── requirements.txt  # atalho: instala o pacote em modo editável
└── requirements-lock.txt  # versões exatas do ambiente de referência
```

**A regra do projeto:** a lógica fica em `src/voc_rag`, testada; os notebooks importam
essa lógica, mostram o código com `mostrar_codigo()` e explicam cada decisão. Cada etapa
também roda pela linha de comando (`python -m voc_rag.<modulo>`), sem uma pasta
`scripts/` repetindo código.

## Como executar

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # instala src/voc_rag em modo editável
cp .env.example .env              # Windows: copy; preencha GOOGLE_API_KEY
pytest                            # 111 testes, sem rede
```

- **Chave do LLM:** gere em [Google AI Studio](https://aistudio.google.com/apikey) e
  coloque em `GOOGLE_API_KEY` no `.env` (que está no `.gitignore`). `LLM_RPM` limita as
  chamadas por minuto, para caber no plano gratuito.
- **Versões exatas:** `requirements-lock.txt` guarda o ambiente em que a base foi gerada
  (Windows, Python 3.14). Para reproduzir igual:
  `pip install -r requirements-lock.txt && pip install -e . --no-deps`.
- **macOS:** o PyTorch e o FAISS trazem cada um sua cópia do OpenMP, e com várias
  threads a primeira busca derruba o processo. O pacote já define
  `OMP_NUM_THREADS=1` no macOS ao ser importado; se usar o FAISS fora dele, exporte a
  variável antes.
- **Windows:** defina `PYTHONIOENCODING=utf-8` no terminal para os acentos saírem certos.

### 1. Preparar a base de conhecimento

```bash
python -m voc_rag                 # baixa os dados, gera a base e o índice
```

- **Token do Kaggle:** em *Kaggle → Settings → API → Create New Token*, salve o
  `kaggle.json` em `~/.kaggle/` (Windows: `C:\Users\<voce>\.kaggle\`). Sem token, baixe
  pelo botão *Download* do Kaggle e descompacte em `data/raw/`.
- **Modelo:** na primeira execução o `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) é
  baixado do Hugging Face. Os embeddings levam de meio minuto a pouco mais de um minuto
  numa CPU comum.
- **Etapas separadas:** `python -m voc_rag.dados`, `python -m voc_rag.preparacao`,
  `python -m voc_rag.indice`. `python -m voc_rag --recalcular` refaz o índice.
- **Rastreabilidade:** o `manifesto.json` guarda modelo, dimensão, hash SHA-256 da base
  e versões. Se a base ou o modelo mudarem, `carregar_indice` recusa o índice antigo. A
  preparação é determinística: os mesmos CSVs geram o mesmo hash, com fim de linha fixo
  para dar o mesmo arquivo no Windows, no Linux e no macOS.

Os dados e os artefatos não vão para o git (licença CC BY-NC-SA 4.0 da Olist).

### 2. Perguntar pela linha de comando

```bash
python -m voc_rag.rag "Quais as reclamações sobre atraso na entrega em SP?"
```

Ou em Python, com o grafo e a resposta tipada:

```python
from voc_rag import demo, rag
documentos, app = demo.carregar_recursos()    # base, índice, LLM e o grafo
estado = rag.responder(app, "O que os clientes elogiam na entrega?")
rag.imprimir(estado["resposta"])     # status, resumo, pontos com review_id e evidências
```

### 3. Demo em Streamlit

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

### 4. Reproduzir a avaliação

A avaliação (notebook `10_avaliacao.ipynb`) foi **rodada uma vez** e tudo ficou
versionado em `avaliacao/`:

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
  `ConfigAvaliacao(modo="ao_vivo", max_perguntas=3, sistemas=["B0", "H5"])` roda três
  perguntas com a chave do `.env` de quem executa (alguns centavos). Antes de rodar, o
  notebook mostra o teto de chamadas. O resultado vai para `artifacts/avaliacao/ao_vivo/`,
  fora do git, e não altera o cache.

O **teste trancado** (20 perguntas) só abre com `abrir_teste=True` e foi rodado uma única
vez, com B0 e o sistema final; o notebook apenas lê o resultado gravado.

## Notebooks: o projeto passo a passo

Cada notebook abre com o contexto, o objetivo e um roteiro de seções, termina cada seção
com uma conclusão tirada da saída e fecha com as decisões e o próximo passo. Lidos em
ordem, contam o projeto inteiro.

| Notebook | O que mostra | Usa LLM? |
|---|---|---|
| `01_base.ipynb` | das reviews brutas aos documentos: seleção, duplicatas, metadados, limpeza leve x normalização, PII | não |
| `02_indice_vetorial.ipynb` | embeddings, cosseno, FAISS `IndexFlatIP`, pré-filtro por metadado, limites da busca vetorial | não |
| `03_busca_hibrida.ipynb` | BM25, fusão com RRF, filtros de negócio, precisão@10 e o sinal de recusa | não |
| `04_guardrail_entrada.ipynb` | schema Pydantic x intenção, PII com validação aritmética, prompt injection, política como código | não |
| `05_roteador.ipynb` | LLM com fallback, `with_structured_output`, filtros extraídos e conferidos com a base | ~7 chamadas |
| `06_multi_query.ipynb` | reformulação da consulta, RRF entre consultas, custo x ganho | ~15 chamadas |
| `07_evidencia_recusa.ipynb` | limiar de similaridade, contexto delimitado, juiz LLM, recusa | ~35 chamadas |
| `08_geracao_resposta.ipynb` | resposta JSON com citação obrigatória, validação das citações, ataques à saída | ~5 chamadas |
| `09_rag_langgraph.ipynb` | `StateGraph`, arestas condicionais, bateria responder/recusar/bloquear, auditoria | ~30 chamadas |
| `10_avaliacao.ipynb` | baseline x hipóteses, gold set, acerto@10/MRR, matriz de recusa, fidelidade, custo, análise de erro, teste trancado | não (lê o cache) |

Os notebooks 01 a 04 e o 10 rodam sem chave. Os notebooks 05 a 09 estão versionados com
as saídas da execução original, para leitura sem custo.

## Exemplos de perguntas e respostas

Respostas reais do RAG final no teste trancado (`avaliacao/respostas/teste_H5.csv`; ids
abreviados aqui só para leitura).

**Pergunta com filtros extraídos pelo roteador**

> **Quais as reclamações de clientes do Rio de Janeiro sobre atraso na entrega?**
> Filtros: `estados=["RJ"]`, `nota_max=2`, `entregue_com_atraso=True`
>
> Nas reviews recuperadas de clientes do Rio de Janeiro, as principais queixas
> concentram-se no descumprimento dos prazos de entrega e na demora excessiva para o
> recebimento dos pedidos. Além disso, há relatos de falta de comunicação ou satisfação
> sobre o atraso por parte da empresa.
>
> - Clientes relatam atraso excessivo e descumprimento do prazo prometido.
>   `[8b592f04…]` `[95e7f613…]` `[72926e52…]` e mais 4
> - Consumidores queixam-se da ausência de suporte ou justificativa sobre o atraso.
>   `[693d2c01…]` `[5845cb7b…]`
> - Cliente reclama de espera superior a 30 dias com envio pelos Correios.
>   `[89a0e383…]`

**Pergunta-agulha: uma única review responde**

> **Alguém recebeu um leitor comum no lugar do leitor de Blu-Ray que comprou?**
>
> Sim. Na review recuperada, um cliente relata que comprou um leitor de Blu-Ray, mas
> recebeu um leitor comum que não lê essa mídia. `[ec80016d…]`

**Recusa e bloqueio**

| Pergunta | Status | Motivo |
|---|---|---|
| O que dizem sobre a experiência de compra no metaverso? | recusada | pergunta fora do escopo das reviews |
| Qual é a capital da França? | recusada | pergunta fora do escopo das reviews |
| Qual o telefone do cliente que reclamou do colchão? | bloqueada | tópico proibido: 'telefone do cliente' |

O bloqueio acontece no guardrail, antes de qualquer chamada ao LLM.

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
| Reviews negativas entre pedidos atrasados | 73% (contra 18% nos entregues no prazo) |

## Avaliação

### Checagem da recuperação

Dez perguntas de analista (`avaliacao.PERGUNTAS_CHECAGEM`), k = 10, nos dados reais
(notebook 03):

| Busca | Precisão@10 |
|---|---|
| Vetorial | 0,85 |
| BM25 | 0,91 |
| Híbrida (RRF) | 0,87 |

O juiz é uma expressão regular por pergunta, o que favorece o BM25: é uma checagem de
sanidade, não a avaliação final. Para recusa, a maior similaridade separou as perguntas
do tema (0,73 a 0,997) das de fora (0,48 a 0,54).

### Avaliação do RAG

Protocolo: baseline congelada (B0: busca vetorial e recusa só pelo prompt), hipóteses
que mudam uma variável por vez, gold set de 60 perguntas em 7 estratos (40 dev, 20 teste
trancado), juízes LLM para relevância e fidelidade. Detalhes e todas as tabelas no
notebook `10_avaliacao.ipynb`.

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
- **H4** foi a lição da rodada: a checagem de evidência zerou a alucinação, mas exigia 2
  evidências, e uma agulha só tem uma review certa. Recusou 7 perguntas com a resposta
  no top-10. A **análise de erro** virou a hipótese **H5** (`MIN_EVIDENCIAS = 1`), que
  recuperou as agulhas sem alucinação nova.
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
intervalos se sobrepõem: o ganho é consistente, mas a amostra é pequena para chamá-lo de
significativo. **Custo total da avaliação:** cerca de R$ 14 no Gemini 3.8 Flash (~750
chamadas); ~77% dos tokens de saída são o raciocínio interno do modelo.

## Decisões de projeto

| Decisão | Por quê |
|---|---|
| Só reviews com texto viram documento; título + mensagem num texto só | review sem texto não é evidência; o título carrega sentido |
| Uma linha por `review_id` | a citação precisa ser única; cópias contariam em dobro |
| `texto` (limpeza leve + PII) para embedding, `texto_lexical` (normalizado, sem acento) para BM25 | embeddings preferem texto natural; BM25 precisa casar "nao" e "não" |
| 1 review = 1 documento, sem chunking | reviews têm no máximo 229 caracteres |
| Metadados fora do texto, usados como pré-filtro | filtro restritivo não esvazia o top-k; chunk enriquecido fica como hipótese |
| MiniLM multilíngue + FAISS `IndexFlatIP` | local, gratuito e entende português; busca exata é rápida com ~42 mil vetores |
| Grava embeddings + manifesto, não o binário do FAISS | o caro é o embedding; o índice flat remonta em milissegundos |
| BM25 montado em memória | frações de segundo sobre a base inteira; sem arquivo para sair de sincronia |
| RRF com k = 60 sobre os 50 primeiros de cada busca | funde por posição, não por pontuações em escalas diferentes |
| Reviews com menos de 3 palavras não viram evidência | com pergunta curta, a vetorial as puxa com similaridade alta e sem sentido |
| Toda saída de LLM em schema Pydantic (`with_structured_output`) | sem parsing de texto livre; campo inventado vira erro (`extra="forbid"`) |
| Saída do LLM conferida com dados reais | categoria inexistente, período fora da base e `review_id` não recuperado são descartados |
| Recusa com duas barreiras: limiar + juiz LLM, mínimo de 1 evidência | o limiar não separa pergunta do domínio sem resposta ("entrega por drone"); mínimo de 2 recusava agulhas (H4 x H5 na avaliação) |
| Avaliação com baseline, uma variável por vez e teste trancado | ganho com causa conhecida; o teste não é usado para ajustar nada |
| Guardrail antes de qualquer LLM; log só com hash | defesa em profundidade, custo zero para entradas bloqueadas, e LGPD |

## Limitações conhecidas

- **Pergunta de uma palavra** ("boleto"): na busca simples, o lado vetorial traz elogios
  genéricos (~0,84). No RAG, o Multi-Query reformula a consulta e o juiz descarta
  evidência genérica (notebooks 06 e 07).
- **Detectores por padrão** (PII, injeção) têm falso positivo e falso negativo; são a
  primeira camada, não a única (notebook 04).
- **Nota x texto:** há reviews em que a nota contradiz o texto ("Não foi entregue o meu
  pedido" com nota 5). `sentimento_nota` vem da nota; um classificador de texto pode
  marcar essas divergências.
- **PII na base:** o mascaramento da base é por regex, sem validação aritmética; pode
  mascarar um número de pedido como telefone (falso positivo aceito). O guardrail da
  pergunta usa a versão com validação.
- **Frequência:** o LLM vê 10 reviews, não a base inteira. A resposta diz o que há
  "nas reviews recuperadas"; contar quantos clientes reclamam de algo é trabalho de
  agregação, não de geração.
- **Avaliação:** os rótulos de relevância e a fidelidade vêm de juízes LLM; o protocolo
  pede revisão humana de uma amostra (o notebook 10 sorteia 20 rótulos para isso), que
  ainda não foi feita. Duas paráfrases (C02 e C07) nenhum sistema acerta: a review certa
  não chega ao top-10. O teste tem só 20 perguntas.
- **Latência e custo:** o H5 leva ~25 s por pergunta (4 chamadas ao LLM, espaçadas pelo
  limitador) contra ~7 s da baseline.

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
