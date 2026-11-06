# Protocolo de avaliação

Como o RAG foi medido, para quem quiser conferir, repetir ou estender a avaliação. As
tabelas completas estão no notebook `10_avaliacao.ipynb`; o resumo dos resultados, no
README.

## Princípios

1. **Gabarito antes da medida.** As 60 perguntas e o resultado esperado de cada uma foram
   escritos antes de rodar qualquer sistema.
2. **Baseline congelada.** B0 é o RAG mais simples que funciona: busca vetorial, top-10 e
   um prompt que manda responder só com as reviews e recusar quando não houver.
3. **Uma variável por vez.** Cada hipótese muda uma coisa em relação à anterior e diz,
   antes de rodar, em que estrato o ganho deve aparecer.
4. **Teste trancado.** 20 perguntas ficam de fora de qualquer ajuste e rodam uma única
   vez, com a baseline e o sistema final.
5. **Análise de erro.** O que um sistema erra vira a próxima hipótese (foi assim que
   nasceu o H5).

## Gold set (`avaliacao/gold_set.csv`)

| Estrato | O que testa | Esperado | Dev | Teste |
|---|---|---|---|---|
| A agregada | tema com centenas de reviews relevantes | respondida; precisão@10 e cobertura de subtemas | 8 | 4 |
| B agulha | um fato que aparece em poucas reviews | respondida; a review-alvo no top-10 | 7 | 3 |
| C paráfrase | a mesma agulha com outras palavras | respondida; a review-alvo no top-10 | 5 | 3 |
| D metadado | filtro na própria pergunta (UF, período, nota, categoria) | respondida; filtros extraídos certos | 7 | 3 |
| E sem evidência | do domínio, mas sem resposta na base | recusada | 5 | 3 |
| F fora de escopo | nada a ver com reviews | recusada | 4 | 2 |
| G adversarial | injeção, PII, dado de terceiro | bloqueada (ou recusada) | 4 | 2 |

A partição dev/teste foi sorteada dentro de cada estrato, com semente fixa.

## Sistemas

| Sistema | Muda | Previsão | Chamadas por pergunta |
|---|---|---|---|
| B0 | baseline: vetorial, recusa no prompt | — | 1 |
| H1 | + busca híbrida (BM25 + vetorial, RRF) | ganha em B | 1 |
| H2 | + Multi-Query | ganha em C e A | 2 |
| H3 | + roteador com filtros | ganha em D | 3 |
| H4 | + guardrail e checagem de evidência (mín. 2) | recusa certa em E, F, G | 4 |
| H5 | H4 com mínimo de 1 evidência | para de recusar agulhas sem alucinar | 4 |

De B0 a H3 as citações saem como o LLM escreveu, para medir quantas seriam inválidas. H4
e H5 são o grafo completo de `rag.py`.

## Métricas

| Métrica | Pergunta que responde | Aplica-se a |
|---|---|---|
| acerto de comportamento | respondeu, recusou ou bloqueou como esperado? | todas |
| acerto@10 e MRR | a review-alvo está no top-10? em que posição? | B, C |
| precisão@10 | que fração do top-10 é relevante? | A, C, D |
| cobertura de subtemas | a resposta cobre os subtemas esperados? | A |
| filtros exatos | o roteador extraiu exatamente os filtros esperados? | D |
| matriz de recusa (precisão, recall, F1) | recusa quando deve, sem recusar à toa? alucina? | todas |
| bloqueou no guardrail | o ataque parou antes de gastar LLM? | G |
| citações válidas | os `review_id` citados estavam no contexto? | respondidas |
| quantificação sem base | a resposta inventa "a maioria" ou percentuais? | respondidas |
| fidelidade | as reviews citadas sustentam cada afirmação? | respondidas |
| latência, chamadas e tokens | quanto custa? | todas |

**Relevância por pooling:** o top-10 de todos os sistemas é juntado e cada review é
rotulada uma vez por um juiz LLM (`avaliacao/respostas/julgamentos.csv`).
**Fidelidade:** outro juiz LLM confere afirmação por afirmação
(`{particao}_{sistema}_fidelidade.csv`). **Teste:** intervalo de confiança de 95% por
bootstrap e comparação pareada pergunta a pergunta.

## Reproduzir

- **Sem custo:** abra o notebook 10 com `ConfigAvaliacao()` (modo `"cache"`). Ele lê as
  respostas versionadas e recalcula todas as métricas; não precisa de chave, da base
  nem do índice.
- **Ao vivo, numa amostra:**
  `ConfigAvaliacao(modo="ao_vivo", max_perguntas=3, sistemas=["B0", "H5"])` roda três
  perguntas com a chave do `.env` (alguns centavos) e grava em
  `artifacts/avaliacao/ao_vivo/`, sem tocar no cache.
- **Teste:** `ConfigAvaliacao(particao="teste")` dá erro sem `abrir_teste=True`, de
  propósito.

A rodada completa custou cerca de R$ 14 no Gemini 3.8 Flash (~750 chamadas, contando os
juízes).

## Pendências

- **Revisão humana dos rótulos.** O notebook 10 sorteia 20 julgamentos de relevância
  para revisão; quem revisar troca `rotulador` para `"humano"` e corrige `relevante` em
  `julgamentos.csv`, e a métrica se recalcula do cache.
- **Paráfrases C02 e C07.** Nenhum sistema acerta: a review certa não chega ao top-10.
  Candidatas a hipótese: mais variações no Multi-Query ou um modelo de embeddings maior.
- **Tamanho do teste.** Com 20 perguntas, os intervalos de B0 e H5 se sobrepõem.
