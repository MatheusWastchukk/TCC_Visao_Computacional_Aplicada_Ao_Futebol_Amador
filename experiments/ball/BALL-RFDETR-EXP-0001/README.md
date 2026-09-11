# BALL-RFDETR-EXP-0001

Experimento detector-only para testar a hipótese de substituir o gerador de
candidatas YOLO por RF-DETR Small, com a classe única `ball`.

## Estado em 07/09/2026

- Pacote isolado: `rfdetr==1.10.1`, PyTorch `2.14.0+cpu`.
- Dataset COCO preparado a partir de `ball_dataset_annotations_v2.csv`:
  733 imagens de treino, 119 de validação, 605 positivas e 247 `NOT_VISIBLE`.
- Smoke test de treino: 20 imagens de treino, 10 de validação, 1 época,
  resolução 384, batch 1, CPU.
- O smoke test concluiu forward, backward, validação e salvou checkpoints.

## Compatibilidade ROCm / AMD — 08/09/2026

O RF-DETR foi instalado no `.venv-rocm` sem substituir o PyTorch
`2.9.1+rocm7.2.1`. O smoke test na AMD Radeon RX 9060 XT completou uma época
em GPU, com dataset de 20 imagens de treino e 10 de validação, sem AMP:

- 384 px, batch 4: 24 s e pico de 3,6 GB de VRAM;
- 512 px, batch 8: 46 s e pico de 9,2 GB de VRAM.

O treinamento é tecnicamente compatível, mas não é totalmente acelerado:
o pareamento Hungarian caiu para SciPy em CPU e a atenção eficiente da AMD é
marcada pelo PyTorch como experimental. A execução também emite um aviso do
`offload-arch` devido ao caminho físico histórico do ambiente conter espaço;
ele não impediu os testes GPU concluídos.

Para 733 imagens de treino e batch 8, a extrapolação conservadora do teste
512 px é de aproximadamente 92 batches, 13–15 minutos por época e 6,5–7,5
horas para 30 épocas, incluindo validação e checkpoints. Esta é uma estimativa,
não uma medição do treino completo.

### Piloto completo interrompido

O piloto de três épocas em 512 px/batch 8 iniciou e completou a época 0 em
5 min 40 s, com `mAP50-95=3,56%` e recall de validação `26,03%`. Durante a
época 1, a telemetria passou a indicar `free_mem=0` e o computador travou,
exigindo reinicialização manual. O resultado da época 0 é evidência de
execução, não resultado comparativo: o treino de três épocas não foi
concluído. Qualquer nova tentativa deve reduzir o batch para 4 e manter
folga de VRAM, aumentando o tempo estimado.

### Candidate recall do checkpoint salvo da época 0

O checkpoint salvo foi avaliado nas 73 imagens `VISIBLE`/`PARTIAL` do mesmo
split `validation`, com IoU 0,50 e os mesmos limiares aplicados ao
`BALL-EXP-0005`. A comparação é preliminar e não é justa como avaliação final:
RF-DETR possui uma época/512 px, enquanto YOLO possui 30 épocas/960 px. Ainda
assim, ela fornece sinal de continuidade.

| Confiança | RF-DETR: candidate recall | RF-DETR: candidatas/frame | YOLO EXP-0005: candidate recall | YOLO: candidatas/frame |
|---:|---:|---:|---:|---:|
| 0,01 | 53,42% | 208,55 | 65,75% | 8,03 |
| 0,03 | 50,68% | 76,08 | 60,27% | 3,53 |
| 0,05 | 45,21% | 34,03 | 56,16% | 2,29 |
| 0,10 | 35,62% | 8,71 | 50,68% | 1,33 |

O sinal é `NO-GO` preliminar: não houve ganho de candidate recall e a carga de
falsas candidatas é muito maior. Junto do travamento em batch 8, isso não
justifica um treino longo de 30 épocas no hardware AMD atual. O experimento
deve ser encerrado salvo nova justificativa para testar uma configuração
substancialmente diferente.

O smoke test confirma viabilidade operacional, mas não mede ganho contra YOLO:
é pequeno demais e tem apenas uma época. O treino comparativo de 30 épocas e a
avaliação de `candidate recall` permanecem pendentes.

## Comandos reproduzíveis

```powershell
.\.venv\Scripts\python.exe experiments\ball\prepare_ball_rfdetr_dataset.py `
  --annotations experiments\ball\ball_dataset_annotations_v2.csv `
  --output experiments\ball\BALL-RFDETR-EXP-0001\dataset_rfdetr

.\.venv\Scripts\python.exe experiments\ball\make_rfdetr_smoke_subset.py `
  --source experiments\ball\BALL-RFDETR-EXP-0001\dataset_rfdetr `
  --output experiments\ball\BALL-RFDETR-EXP-0001\dataset_smoke
```

O ambiente `.venv-rfdetr` e os checkpoints/datasets gerados são locais e
ignorados pelo Git.

Para ROCm, execute o Python pelo atalho físico `C:\tcc_cv` e configure os
caches MIOpen conforme `docs/configuracao-gpu-amd-directml.md`. Use
`device="cuda"` (não `"cuda:0"`): a versão 1.10.1 do RF-DETR possui um bug
interno ao receber o índice explícito do dispositivo.
