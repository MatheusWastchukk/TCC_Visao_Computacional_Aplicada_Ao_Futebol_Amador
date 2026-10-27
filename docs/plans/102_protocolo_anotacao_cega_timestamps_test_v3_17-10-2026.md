# Protocolo 102 — Anotação cega dos timestamps do catálogo TEST V3

**Data do protocolo:** 17/10/2026<br>
**Aplicável a:** 9 partidas TEST e 57 registros do catálogo-base.<br>
**Pré-condição:** freeze V3 já emitido.<br>
**Objetivo:** fixar timestamps humanos independentes antes de qualquer inferência ou consulta às saídas V3.

## Cegamento

Durante a anotação, usar somente o vídeo original indicado no split manifest e o ID/ordem do gol catalogado. Não abrir predições, escores, rankings, shortlist, overlays, clipes selecionados, CSVs de candidatos ou relatórios de cobertura. O anotador pode ver e navegar pelo vídeo completo para localizar o evento. Nenhum artefato de TEST será passado ao ranker nesta etapa.

## Convenção de timestamp

- O timestamp é o tempo, em segundos desde o primeiro frame do arquivo de vídeo original, do frame mais próximo ao instante visual em que a bola cruza o plano da linha de gol para dentro da meta.
- Registrar o frame exibido mais próximo desse instante; a ferramenta grava segundos com precisão decimal de milissegundo e também `HH:MM:SS` arredondado para o segundo.
- Se o cruzamento não estiver visível, mas houver evidência visual inequívoca no vídeo que permita localizar o evento, registrar o frame mais próximo da primeira evidência visual inequívoca e explicar a convenção usada em `notes`.
- Se o vídeo não permitir localizar o evento com segurança, usar `not_evaluable`; não estimar por placar, narração, comentários ou saída do modelo.
- Se o item catalogado não corresponder a um gol identificável naquela partida, usar `catalog_discrepancy` e explicar o motivo em `notes`. Não transformar discrepância em `NO_GOAL`.
- A avaliação congelada usa o timestamp como ponto exato: `covered=true` quando `event_timestamp_seconds` pertence à união fechada de intervalos selecionados, conforme o código Nested (`start <= t <= end`). Este protocolo não acrescenta uma tolerância pós-hoc.

## Uso do anotador

Arquivo de entrada gerado após o freeze:

`experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/goal_catalog_timestamp_manifest_v1.csv`

Salvar o trabalho no arquivo de saída separado, para preservar o template:

`experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/goal_catalog_timestamp_manifest_completed_v1.csv`

Atalhos mostrados na janela:

- `Espaço`: iniciar/parar reprodução.
- `A` / `D`: recuar/avançar 5 segundos.
- `J` / `L`: recuar/avançar 30 segundos.
- `G`: registrar o frame atual como `completed`.
- `U`: marcar `not_evaluable`.
- `X`: marcar `catalog_discrepancy`.
- `W`: salvar progresso; `Q` ou `Esc`: salvar e encerrar.

Os estados suportados pelo anotador são `pending`, `completed`, `not_evaluable` e `catalog_discrepancy`. Ao usar `U` ou `X`, anotar manualmente o motivo na coluna `notes` depois de salvar. Não criar estados novos na planilha.

## Validação e congelamento do catálogo

Antes de usar os timestamps na avaliação:

1. Confirmar 57 IDs únicos, os nove `match_id` esperados e ausência de `pending`.
2. Confirmar que timestamps `completed` são numéricos, não negativos e menores ou iguais à duração do vídeo correspondente.
3. Confirmar consistência entre `event_timestamp_seconds` e `event_timestamp_hhmmss`, IDs duplicados, estados e notas de itens não avaliáveis/divergentes.
4. Registrar por partida as contagens `completed`, `not_evaluable` e `catalog_discrepancy`; o denominador principal será a soma dos registros `completed`, sem assumir previamente que seja 57.
5. Rever novamente, ainda sem saídas de IA, todos os registros sinalizados como incertos. Se houver segundo revisor disponível, ele deve permanecer cego e desacordos/resolução devem ser documentados.
6. Salvar SHA-256 do CSV final e um log com operador, data, duração/FPS por vídeo, contagens, revisões e a confirmação de que nenhuma saída de IA foi aberta.
7. Não editar o CSV depois de calculado o hash. Qualquer correção antes da inferência exige registrar a versão corrigida e recalcular o hash; depois da primeira inferência, o catálogo não pode ser alterado para beneficiar uma política.

## Limite desta etapa

Este protocolo não executa ranking nem mede cobertura. O relatório 101 identificou que ainda é preciso criar e auditar o executor experimental que carrega os modos V3; a anotação cega pode avançar enquanto esse trabalho é preparado. A avaliação preditiva só começa quando o CSV humano estiver congelado e o gate P0-A estiver aprovado.
