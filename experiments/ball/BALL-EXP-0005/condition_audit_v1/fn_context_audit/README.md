# Fila de revisão de contexto dos falsos negativos

Esta pasta contém a fila dos 32 falsos negativos do `BALL-EXP-0005` em
confiança 0,05, derivados de `condition_frame_metrics.csv`. Os overlays exibem
a caixa anotada em verde; eles não exibem uma predição, porque não houve
candidata com IoU maior ou igual a 0,50 para a bola anotada.

O arquivo `fn_context_review_queue.csv` inicia cada item como
`FALSO_NEGATIVO` e `context_status=pending`. Os arquivos `.jpg` são artefatos
locais ignorados pelo Git; a fila CSV pode ser versionada depois da revisão.

## Resultado da revisão concluída em 08/09/2026

Todos os 32 itens foram classificados. A distribuição está em
`fn_context_summary.csv`:

| Contexto principal | Falsos negativos | Proporção |
|---|---:|---:|
| Borrada/baixa qualidade | 11 | 34,38% |
| Muito pequena | 6 | 18,75% |
| Oculta por jogador | 6 | 18,75% |
| Cobrança lateral | 3 | 9,38% |
| Fora do enquadramento | 2 | 6,25% |
| Mãos do goleiro | 2 | 6,25% |
| Outro | 2 | 6,25% |

Os contextos descrevem os falsos negativos do YOLO em confiança 0,05; não
estimam a distribuição de todos os quadros do conjunto. Em `OUTRO`,
`match_14_ball_009` não teve candidata mesmo em 0,01, enquanto
`match_26_ball_010` teve candidata de confiança 0,534 com IoU 0,497 — abaixo
do critério de 0,50 por 0,003, portanto quase acerto de localização.

## Revisão humana

Execute o anotador a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe experiments\ball\annotate_ball_error_context.py `
  --input experiments\ball\BALL-EXP-0005\condition_audit_v1\fn_context_audit\fn_context_review_queue.csv `
  --pending-only
```

Teclas: `1` mãos do goleiro, `2` cobrança lateral, `3` ocultação por jogador,
`4` muito pequena, `5` borrada/baixa qualidade, `6` fora do enquadramento,
`7` outro e `8` inconclusivo. `Q` salva o progresso e encerra.

Os rótulos são uma classificação visual de causa principal. A revisão foi
concluída com todos os itens em `context_status=complete`; os totais por
contexto são evidência anotada, limitada a esta fila de falsos negativos.
