# Relatório 99 — Seleção LOMO dos modos sem resgate

**Data:** 14/10/2026<br>
**Execução:** `REPORT98-NO-RESCUE-MODES-LOMO-V3`<br>
**População:** 25 partidas TRAIN, 5.742 grupos candidatos e 152 gols catalogados.

## Decisão

Foram mantidos três modos operacionais porque a seleção encontrou configurações distintas e com cobertura crescente dentro dos limites de minutos. Os três usam `RRF4_NET` sem resgate CORE; variam janela e K. Isso elimina a duplicação que havia entre `BALANCED_V2` e `HIGH_RECALL_V2`.

| Modo V3 | Limite | Método | Janela | K | Cobertura LOMO usada na seleção | Média de minutos únicos | Máximo por partida |
|---|---:|---|---:|---:|---:|---:|---:|
| FAST_V3 | 30 min | RRF4_NET | 30 s | 60 | 111/152 (73,0%) | 29,38 | 30,00 |
| BALANCED_V3 | 40 min | RRF4_NET | 40 s | 60 | 126/152 (82,9%) | 39,17 | 40,00 |
| HIGH_RECALL_V3 | 58 min | RRF4_NET | 60 s | 55 | 133/152 (87,5%) | 52,86 | 54,61 |

As configurações são diferentes e cada faixa mais lenta aumentou a cobertura agregada de seleção. O terceiro modo ficou abaixo do teto de 58 minutos, com K=55; elevar o K não foi necessário para a configuração selecionada.

## Como a seleção foi feita

O seletor refez previsões externas por partida para os 72 pontos congelados da grade NO_RESCUE: 3 métodos × 4 janelas × 6 valores de K. Em cada fold, a partida avaliada ficou de fora dos modelos temporal, visual, net-motion e fusão. Os dados de desenvolvimento da fusão também foram formados por escores base cross-fitted, excluindo a partida externa e a interna.

Para cada orçamento, só foram elegíveis configurações cujo máximo de minutos únicos entre as 25 partidas não excedeu o limite. A escolha maximizou gols cobertos; desempates usam menor média de minutos, menos segmentos e a ordem de complexidade congelada. Um modo mais lento só foi mantido quando era distinto e aumentava a cobertura em relação ao tier anterior.

## Interpretação e limites

As coberturas 111/152, 126/152 e 133/152 são resultados da mesma LOMO usada para escolher entre 72 configurações. Portanto, descrevem o critério de seleção e não são estimativas imparciais de desempenho. A evidência principal continua sendo o Nested LOMO 25/25 do relatório 95, que estima a regra de seleção por orçamento; o experimento e seus resultados não foram alterados por este trabalho.

O relatório 95 mostrou que `NO_RESCUE` superou `CORE_RESCUE` em cobertura pontual nos cinco orçamentos avaliados. Isso motivou restringir a seleção V3 a `TEMPORAL`, `RRF3` e `RRF4_NET`. A seleção atual não abre novo método, não retuna o pipeline congelado e não transforma estes números de TRAIN em resultado de VALIDATION ou TEST.

## Freeze e TEST

O freeze V3 supersede o manifesto V2 e registra somente as três configurações acima. Os pesos operacionais são os mesmos quatro modelos full-TRAIN registrados no pacote V2; este trabalho não os retreinou. TEST permanece fechado. Há 9 vídeos e 57 gols registrados, mas os timestamps ainda precisam ser anotados cegamente nos vídeos originais após o freeze V3. Depois da anotação e validação do catálogo, executar uma única avaliação, sem retuning nem escolha de vencedor pelo resultado de TEST.

## Artefatos

- Protocolo prévio: [plano 98](98_plano_selecao_modos_lomo_no_rescue_v3_14-10-2026.md).
- Grade e cobertura por partida: `experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/fixed_policy_lomo_by_match_v3.csv`.
- Agregação das 72 configurações: `experiments/ball/REPORT98-NO-RESCUE-MODES-LOMO-V3/fixed_policy_lomo_summary_v3.csv`.
- Configuração operacional e hashes dos modelos: `final_modes_v3.json`, `run_manifest_v3.json` e `model_manifest_v3.json` no mesmo diretório.
- Freeze: `experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v3.json`.
- Anotação cega de TEST condicionada ao freeze: `prepare_test_goal_catalog_manifest_v3.py` e `annotate_test_goal_catalog_timestamps_v3.py`.
