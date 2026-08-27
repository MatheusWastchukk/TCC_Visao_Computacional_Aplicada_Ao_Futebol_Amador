# Análise do BALL-DATASET-V1 e ajustes do plano

## Situação da anotação

O lote de 620 frames foi concluído, sem amostras pendentes e com caixas preenchidas em todos os frames `VISIBLE` e `PARTIAL`.

| Visibilidade | Total | Percentual |
|---|---:|---:|
| `VISIBLE` | 395 | 63,71% |
| `PARTIAL` | 25 | 4,03% |
| `NOT_VISIBLE` | 184 | 29,68% |
| `UNCERTAIN` | 16 | 2,58% |
| Total | 620 | 100% |

## Interpretação

A dificuldade observada é esperada para vídeos de futebol amador: a bola pode estar atrás de jogadoras, borrada, acima do enquadramento, nas mãos da goleira, distante ou invisível pela qualidade da imagem.

`NOT_VISIBLE` não é falha do detector. Esses frames serão mantidos como negativos de contexto, com arquivo YOLO vazio, mas não serão usados para calcular recall da bola.

`UNCERTAIN` será excluído do treino e da validação principal até haver um protocolo específico para casos ambíguos.

## Distribuição por split

- `train`: 347 frames localizáveis (`VISIBLE` + `PARTIAL`), 138 `NOT_VISIBLE` e 15 `UNCERTAIN`;
- `validation`: 73 frames localizáveis, 46 `NOT_VISIBLE` e 1 `UNCERTAIN`;
- `test`: nenhum frame utilizado.

Entre frames localizáveis, a distribuição é mais difícil no XBotGo: 47 de 94 frames são `VISIBLE` ou `PARTIAL` (50,00%). No Veo/derivados, são 373 de 526 (70,91%). O sistema de captura deve permanecer registrado nas métricas.

## Ajustes metodológicos

1. O baseline inicial deve responder se a bola é localizável nos frames avaliáveis, não se ela aparece em todos os frames.
2. A métrica principal de recall deve usar `VISIBLE` e `PARTIAL`.
3. `NOT_VISIBLE` deve ser reportado como indisponibilidade visual e pode ser usado como contexto negativo.
4. `UNCERTAIN` deve ficar fora da métrica principal.
5. A validação deve ser separada por partida e por sistema de captura.
6. Se o baseline falhar principalmente em bolas pequenas, comparar resolução de forma controlada antes de trocar arquitetura.

## Próximo experimento

Preparar o formato YOLO e executar `BALL-EXP-0001` com YOLO11n, mantendo o dataset espacial e temporal congelados.
