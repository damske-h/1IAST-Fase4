# Pipeline RAG: como uma pergunta vira resposta

Nota técnica do grafo em `src/voc_rag/rag.py`. O passo a passo com exemplos está nos
notebooks 04 a 09; aqui fica a referência rápida: nós, contratos, parâmetros e onde cada
requisito é atendido.

## O grafo

```
    guardrail ─bloqueada──────────────────────────────────────────────┐
        │                                                             │
      rotear ─fora_de_escopo──────────────────────┐                   │
        │                                         ▼                   ▼
    recuperar ─> checar ─sem evidência─────────> recusar ─────────> auditar ─> FIM
                    │                                                 ▲
                    └─────────────> gerar ────────────────────────────┘
```

| Nó | Usa LLM? | Entra | Sai | Pode encerrar como |
|---|---|---|---|---|
| `guardrail` | não | texto + filtros explícitos | `AvaliacaoEntrada` (veredito, motivos, tipos de PII, hash) | bloqueada |
| `rotear` | sim (1) | `Pergunta` validada | `Roteamento`: intenção, consulta, `FiltrosBusca` | recusada (fora de escopo) |
| `recuperar` | sim (1) | consulta + filtros | consultas do Multi-Query e top-10 fundido por RRF | — |
| `checar` | sim (0 ou 1) | top-10 | `Decisao`: ids aprovados e motivo | recusada (sem evidência) |
| `gerar` | sim (1) | evidências aprovadas | `RespostaGerada` com citações validadas e PII mascarada | recusada (nenhuma citação válida) |
| `recusar` | não | motivo | `RespostaRAG` de recusa | recusada |
| `auditar` | não | estado final | `RespostaRAG` completa + uma linha no log | — |

Uma pergunta respondida custa 4 chamadas ao LLM; uma bloqueada, nenhuma. O `juiz` do
`checar` só é chamado quando alguma review passa no limiar de similaridade.

## Contratos (Pydantic)

| Schema | Papel |
|---|---|
| `Pergunta`, `FiltrosBusca` | entrada validada: 3 a 500 caracteres, UF, notas, período `AAAA-MM`, categoria; campo desconhecido é erro |
| `Roteamento` | saída do roteador; as `description` dos campos vão ao LLM como instrução |
| `Variacoes` | reformulações do Multi-Query, sem parsing de texto livre |
| `JulgamentoEvidencia` | ids que o juiz considera relevantes e o motivo |
| `RespostaGerada`, `Ponto` | saída do gerador: cada `Ponto` exige ao menos um `review_id` |
| `RespostaRAG` | contrato final: `status`, `resumo`, `pontos`, `evidencias`, `motivo`, `consultas`, `filtros`, `avisos` |

Toda saída do LLM ainda é conferida com dados reais: categoria fora da base e período
fora de 2016-10 a 2018-08 são descartados com aviso; `review_id` citado fora das
evidências é removido; afirmação que fica sem citação é descartada.

## Parâmetros (`src/voc_rag/config.py`)

| Parâmetro | Valor | De onde veio |
|---|---|---|
| `MODELO_EMBEDDINGS` | `paraphrase-multilingual-MiniLM-L12-v2` | local, gratuito, entende português, 384 dimensões |
| `PROFUNDIDADE_FUSAO` | 50 | candidatos de cada busca que entram no RRF |
| `RRF_K` | 60 | valor do artigo original do RRF |
| `MIN_PALAVRAS_EVIDENCIA` | 3 | reviews de 1 ou 2 palavras eram puxadas por perguntas curtas (notebook 03) |
| `LIMIAR_EVIDENCIA` | 0,55 | separou perguntas do tema (≥ 0,73) e de fora (≤ 0,54) no notebook 03 |
| `QTD_VARIACOES` | 3 | reformulações pedidas ao LLM, além da consulta original |
| `K_CONTEXTO` | 10 | reviews que vão para o juiz e para o gerador |
| `MIN_EVIDENCIAS` | 1 | era 2; a avaliação mostrou que 2 recusava agulhas (H4 x H5) |
| `LLM_RPM` (`.env`) | 8 | chamadas por minuto, para caber no plano gratuito do Gemini |

## Requisito por requisito

| Requisito | Onde é atendido |
|---|---|
| Responder com evidências (`review_id`) | `geracao`: schema com citação obrigatória + `validar_citacoes` |
| Recusar quando não há informação | roteador (`fora_de_escopo`) e `evidencia.checar_evidencia` (limiar + juiz) |
| Filtros por metadado | roteador → `FiltrosBusca` → pré-filtro nas duas buscas do `Buscador` |
| Rastreabilidade | `trilha` do estado, `RespostaRAG` com evidências e o log de auditoria |
| Segurança e LGPD | guardrail de entrada, contexto delimitado, PII mascarada na base e na saída, log só com hash (ver [seguranca.md](seguranca.md)) |
