# Pacote de nuvem — BALL-RFDETR-EXP-0001

Pacote autocontido para um único teste controlado em GPU NVIDIA. O objetivo é
confirmar ou refutar o sinal `NO-GO` preliminar obtido localmente em AMD/ROCm;
não é uma nova frente de pesquisa ou uma troca automática de detector.

## Conteúdo a enviar

```text
cloud_bundle_v1/
├── dataset_rfdetr/                 # 733 train + 119 valid, COCO
├── metadata/ball_dataset_annotations_v2.csv
├── reference/yolo_exp0005_candidate_recall_summary.csv
├── reference/yolo_exp0005_condition_summary.csv
├── scripts/train_rfdetr_cloud.py
├── scripts/evaluate_rfdetr_cloud.py
├── requirements.txt
├── cloud_test_config.json
└── run_cloud_test.sh
```

Não contém `test`, vídeos, ambientes virtuais, caches, checkpoints locais ou
outros experimentos. O split permanece em `train` e `validation`.

## Preparação na instância NVIDIA

Use Ubuntu 24.04 ou equivalente com driver NVIDIA funcional. No diretório para
o qual o pacote foi enviado:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
chmod +x run_cloud_test.sh
./run_cloud_test.sh 2>&1 | tee output/cloud_run.log
```

O último comando cria `output/train/` e `output/candidate_recall/`. Verifique
se a linha CUDA informa a GPU NVIDIA esperada antes de iniciar o treino.

## Protocolo congelado

- RF-DETR Small, uma classe `ball`, resolução 512, 30 épocas;
- batch 4 com acumulação de gradiente 4 e AMP habilitado;
- mesmo dataset COCO, divisão e limiares da comparação local;
- candidate recall em 73 quadros `VISIBLE`/`PARTIAL` com IoU 0,50;
- média de candidatas nos 46 quadros `NOT_VISIBLE`;
- estratos por visibilidade e tamanho da caixa anotada;
- comparação com a referência em `reference/`.

Não use o split `test`. Não inicie tracker, ByteTrack, OC-SORT, SAM3,
MediaPipe ou VLM nesta instância, independentemente do resultado do treino.

## Resultado a baixar antes de destruir a instância

Baixe a pasta `output/` inteira, incluindo:

```text
output/train/cloud_training_manifest.json
output/train/checkpoint_best_ema.pth
output/train/metrics.csv
output/train/training_config.json
output/candidate_recall/candidate_recall_frames.csv
output/candidate_recall/candidate_recall_summary.csv
output/cloud_run.log
```

Se o nome do checkpoint salvo divergir, execute a avaliação separadamente:

```bash
python scripts/evaluate_rfdetr_cloud.py \
  --checkpoint output/train/ARQUIVO_DO_CHECKPOINT.pth \
  --annotations metadata/ball_dataset_annotations_v2.csv \
  --images dataset_rfdetr/valid/images \
  --output output/candidate_recall \
  --device cuda
```

Pare/destrua a instância só depois de baixar `output/`; o armazenamento de
instâncias GPU pode não persistir após a destruição.
