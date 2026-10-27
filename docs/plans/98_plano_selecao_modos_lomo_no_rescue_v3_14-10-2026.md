# Plano 98 — Seleção LOMO dos modos operacionais sem resgate

**Data:** 14/10/2026<br>
**Escopo:** TRAIN, somente; nenhum acesso a VALIDATION ou inferência em TEST.

## Motivo

O Nested LOMO full-pool 25/25 preserva a estimativa principal da regra de seleção por orçamento. Nesse resultado, `NO_RESCUE` superou `CORE_RESCUE` em cobertura pontual nos cinco orçamentos (30, 40, 45, 50 e 58 minutos). Além disso, os modos `BALANCED_V2` e `HIGH_RECALL_V2` usam exatamente a mesma configuração. A seleção V2 havia escolhido configurações com métricas de ajuste no TRAIN completo, então a diferenciação dos modos será refeita usando previsões realmente fora da partida avaliada.

## Protocolo congelado antes da execução

- Manter intactos o experimento Nested 25/25, seu protocolo e seus resultados.
- Usar as mesmas 25 partidas TRAIN, 5.742 grupos candidatos, 152 gols catalogados, gap de 45 s e grade congelada de janelas e K do Nested.
- Comparar somente `TEMPORAL`, `RRF3` e `RRF4_NET`, sem resgate CORE, em todas as 72 configurações (4 janelas × 6 valores de K × 3 métodos).
- Para cada partida externa, excluir seus dados do ajuste dos modelos temporal, visual, net-motion e fusão. Os dados de desenvolvimento para fusão devem ter escores base cross-fitted por partida interna, excluindo simultaneamente a partida externa e a interna.
- Consolidar cobertura e minutos a partir das 25 partidas externas. Uma configuração é elegível para um orçamento se o máximo de minutos únicos entre as partidas externas não exceder o limite (30, 40 ou 58 min/partida).
- Selecionar cada tier maximizando gols cobertos no agregado. Desempatar por menor média de minutos únicos, menos segmentos e ordem de complexidade já definida no seletor Nested.
- Só emitir um tier mais lento quando for uma política diferente e cobrir estritamente mais gols que o tier anterior efetivamente selecionado. Modos duplicados ou tiers sem ganho serão omitidos.
- As métricas agregadas dessas 72 configurações servem para seleção e não são estimativas independentes de desempenho: a mesma validação cruzada é usada para escolher. A avaliação Nested 25/25 permanece a evidência principal da regra de seleção por orçamento.

## Critérios de integridade

- Confirmar 25 partidas e 152 gols no catálogo TRAIN, com 25 linhas por configuração e 1.800 linhas OOF no total.
- Confirmar elegibilidade pelo máximo de minutos por partida e ausência de configuração duplicada nos modos emitidos.
- Registrar entradas, hashes, script, regra de seleção e resultados compactos.
- Gerar um freeze V3 que supersede V2 e mantém TEST fechado até a anotação cega dos timestamps dos gols de TEST. Após o freeze, executar a avaliação de TEST uma única vez, sem retuning.

## Decisões permitidas

O resultado pode sustentar um, dois ou três modos operacionais. Não haverá obrigação de manter três opções se os dados não mostrarem tiers distintos com maior cobertura. Nenhuma configuração será promovida com base em desempenho medido em TEST.
