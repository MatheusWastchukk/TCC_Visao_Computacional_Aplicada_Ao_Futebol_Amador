# Auditoria de condições do candidate recall — BALL-EXP-0005

Esta auditoria reutiliza as previsões por quadro já calculadas para as 73
imagens `VISIBLE`/`PARTIAL` do split `validation`, com IoU 0,50. Não houve
novo treino nem nova inferência.

## Condições auditadas automaticamente

- visibilidade anotada (`VISIBLE` ou `PARTIAL`);
- maior lado da caixa da bola: até 16 px, 17–24 px, 25–32 px ou acima de 32 px;
- partida e domínio de captura.

As faixas de tamanho foram definidas após inspecionar a distribuição observada
no conjunto: 5, 32, 18 e 18 quadros, respectivamente. Elas descrevem a escala
da anotação; não são rótulos qualitativos manuais.

## Resultado de referência: confiança 0,05

| Condição | Quadros | Acertos | Candidate recall | Candidatas/quadro |
|---|---:|---:|---:|---:|
| Total | 73 | 41 | 56,16% | 2,29 |
| Visível | 67 | 40 | 59,70% | 2,33 |
| Parcial | 6 | 1 | 16,67% | 1,83 |
| Até 16 px | 5 | 0 | 0,00% | 1,20 |
| 17–24 px | 32 | 19 | 59,38% | 2,72 |
| 25–32 px | 18 | 8 | 44,44% | 1,72 |
| Acima de 32 px | 18 | 14 | 77,78% | 2,39 |

O estrato de até 16 px (n=5) e o de visibilidade parcial (n=6) são pequenos:
o resultado é um sinal para orientar melhoria e anotação, não uma estimativa
estatística definitiva. Nos quatro limiares avaliados, o pior estrato é o de
bola até 16 px; a visibilidade parcial também permanece abaixo da visível.

## Limites e próximo uso

Proximidade de jogador, bola no ar, proximidade do gol e desfoque não podem
ser inferidos fielmente pelas caixas atuais. Eles foram registrados em
`condition_audit_metadata.json` como pendentes de revisão visual manual, sem
criar rótulos artificiais. Esta auditoria não mede falso-positivo em quadros
`NOT_VISIBLE`, pois o CSV de entrada foi produzido apenas para bolas anotadas.

## Reprodução

```powershell
.\.venv\Scripts\python.exe experiments\ball\analyze_ball_candidate_recall_conditions.py `
  --annotations experiments\ball\ball_dataset_annotations_v2.csv `
  --predictions experiments\ball\BALL-RFDETR-EXP-0001\pilot_rocm_r512_e3_b8\baseline_yolo_exp0005_candidate_recall\candidate_recall_frames.csv `
  --output experiments\ball\BALL-EXP-0005\condition_audit_v1
```

Os dados por quadro estão em `condition_frame_metrics.csv`; os agrupamentos,
incluindo todos os limiares, em `condition_summary.csv`.
