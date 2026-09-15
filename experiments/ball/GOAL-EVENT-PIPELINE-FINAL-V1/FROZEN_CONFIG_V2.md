# Configuração congelada V2 — mineração de eventos candidatos a gol

**Data:** 13/09/2026
**Estado:** congelada após as três hipóteses do relatório 43
**Escopo de treino:** somente `TRAIN`
**Conjunto `TEST`:** fechado

## Decisão

O ranker operacional permanece o `RandomForestClassifier` temporal da configuração
V1, treinado em todos os grupos de `TRAIN` e serializado como
`temporal_ranker.joblib` (artefato reproduzível e ignorado pelo Git).

O último ciclo não autorizou troca de modelo:

- hard negatives: NO-GO em LOMO;
- alvo temporal graduado: NO-GO em LOMO;
- atributos de trajetória: ganho em LOMO, mas sem ganho claro e estável na
  confirmação única em `VALIDATION`.

## Contrato operacional

1. Minerar sinais geométricos por partida completa com os detectores congelados.
2. Formar grupos usando `merge_gap_seconds=3`, pelo menos dois hits e 45 s de
   separação entre candidatos selecionados.
3. Extrair os 36 atributos temporais V1.
4. Aplicar o Random Forest temporal V1.
5. Retornar, em ordem de score, no máximo **12** clipes de 30 s (10 s antes e
   20 s depois do timestamp central).
6. Exibir o score do verificador visual somente como evidência auxiliar; ele não
   filtra, não altera o ranking principal e não define rótulo binário.

O limite de 12 é uma decisão de produto baseada na meta operacional do relatório
43: preserva uma lista curta, com aproximadamente seis minutos de revisão, sem
alegar que o sistema detecta todos os gols.

## Evidência de desenvolvimento e confirmação

| Configuração | TRAIN LOMO @8 | TRAIN LOMO @12 | VALIDATION @8 | VALIDATION @12 | Decisão |
|---|---:|---:|---:|---:|---|
| Random Forest temporal V1 | 30/152 | 36/152 | 12/51 | 13/51 | manter |
| Hard negatives | 21/152 | 27/152 | — | — | NO-GO |
| Target graduado | 17/152 | 23/152 | — | — | NO-GO |
| Trajetória enriquecida | 33/152 | 38/152 | 11/51 | 14/51 | NO-GO |

Os valores de `VALIDATION` são confirmação de desenvolvimento e não resultado
final. A configuração não pode sofrer retreinamento, ajuste de hiperparâmetro,
mudança de janela, novo limiar ou seleção baseada em `TEST`.

## Protocolo para TEST

Somente após a integração do backend e a verificação de reprodutibilidade:

1. Executar uma única vez a mineração com esta configuração congelada.
2. Comparar os candidatos com os timestamps de referência sem alterar o pipeline.
3. Aceitar os resultados como avaliação final, inclusive se forem inferiores ao
   esperado.
