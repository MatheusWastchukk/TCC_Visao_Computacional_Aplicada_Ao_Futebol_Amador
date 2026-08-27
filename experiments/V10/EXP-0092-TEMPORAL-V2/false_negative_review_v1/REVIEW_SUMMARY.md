# Revisão dos falsos negativos temporais

A revisão visual dos 12 intervalos classificados como FN mostrou que o conjunto não representa somente falhas puras de persistência. Os casos incluem estruturas pequenas/parciais, estruturas secundárias, intervalos heterogêneos e confusões espaciais com postes ao fundo.

## Classificação consolidada

- 7 casos mantidos como FN de estrutura pequena, parcial, secundária ou múltipla;
- 1 caso encaminhado para auditoria espacial por confusão com postes ao fundo;
- 1 caso misto, encaminhado para revisão conjunta da anotação e da localização espacial;
- 2 casos encaminhados para revisão da fronteira do intervalo, pois um dos frames revisados não mostra estrutura;
- 1 caso mantido como FN de estrutura secundária com detecção de baixa taxa.

O detalhamento está em `false_negative_review_summary_v1.csv`. A métrica original do V2 permanece preservada; esta classificação é uma camada de auditoria e não altera o ground truth sem uma nova decisão explícita.

## Decisão metodológica

Não alterar ainda o limiar temporal nem retreinar o detector espacial. Os casos confirmam a necessidade de tratar estruturas pequenas/secundárias e de separar confusão espacial de perda temporal.

Próxima frente: preparar o conjunto de anotação da bola em partidas de `train`, mantendo o `test` fechado. Depois, os casos espaciais poderão ser usados como auditoria direcionada, sem contaminar a validação temporal original.
