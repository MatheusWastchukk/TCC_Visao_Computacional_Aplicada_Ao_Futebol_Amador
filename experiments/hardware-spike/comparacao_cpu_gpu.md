# Comparacao CPU vs GPU

Data: 29/07/2026

## Resultado atual

| Ambiente | Backend | Status | Evidencia |
|---|---|---|---|
| Computador pessoal | CPU / PyTorch 2.13.0+cpu | Executado | `cpu_Video_Gol_1.json`, `cpu_Video_Nao_Gol_1.json`, `cpu_Video_Nao_Gol_2.json`, `cpu_Video_Nao_Gol_3.json`, `cpu_Video_Nao_Gol_4.json` |
| Computador pessoal | CUDA | Bloqueado | `torch.cuda.is_available() == False` |
| Computador pessoal | DirectML | Executado em `.venv-dml` | `dml_Video_Gol_1.json`, `dml_Video_Nao_Gol_1.json`, `dml_Video_Nao_Gol_2.json`, `dml_Video_Nao_Gol_3.json`, `dml_Video_Nao_Gol_4.json` |
| Computador pessoal | ONNX Runtime GPU | Bloqueado | Providers disponiveis: `AzureExecutionProvider`, `CPUExecutionProvider` |
| Universidade | NVIDIA CUDA / RTX 3060 | Pendente | Requer execucao no computador da universidade |

## CPU com clipes reais

| Clipe | Frames | Tempo total (s) | FPS | Delta memoria (MB) |
|---|---:|---:|---:|---:|
| `Video_Gol_1.mp4` | 100 | 1.6795 | 59.54 | 18.05 |
| `Video_Nao_Gol_1.mp4` | 100 | 1.6978 | 58.90 | 17.82 |
| `Video_Nao_Gol_2.mp4` | 100 | 1.6134 | 61.98 | 17.40 |
| `Video_Nao_Gol_3.mp4` | 100 | 1.7021 | 58.75 | 17.34 |
| `Video_Nao_Gol_4.mp4` | 100 | 0.7825 | 127.79 | 17.25 |

Resumo CPU:

| Metrica | Valor |
|---|---:|
| Clipes reais testados | 5 |
| Frames reais processados | 500 |
| Tempo medio por 100 frames | 1.4951 s |
| FPS medio | 73.39 |
| Delta medio de memoria | 17.57 MB |

## Tentativa GPU

Comando executado:

```powershell
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Gol_1.mp4 --device cuda --frames 100 --output experiments\hardware-spike\gpu_Video_Gol_1.json
```

Resultado:

```text
RuntimeError: Device 'cuda' requested, but CUDA is not available
```

Verificacoes complementares:

```text
torch 2.13.0+cpu
cuda_available False
cuda_device_count 0
torch_directml False
onnxruntime_providers ['AzureExecutionProvider', 'CPUExecutionProvider']
```

## DirectML com clipes reais

Foi criado um ambiente separado `.venv-dml` para preservar a `.venv` principal. Dependencias registradas em `requirements-directml.txt`.

Validacao do dispositivo:

```text
torch 2.4.1+cpu
dml_device privateuseone:0
device_count 1
```

| Clipe | Frames | Tempo total (s) | FPS | Delta memoria (MB) |
|---|---:|---:|---:|---:|
| `Video_Gol_1.mp4` | 100 | 1.7484 | 57.20 | 56.05 |
| `Video_Nao_Gol_1.mp4` | 100 | 1.7413 | 57.43 | 55.75 |
| `Video_Nao_Gol_2.mp4` | 100 | 1.6683 | 59.94 | 55.39 |
| `Video_Nao_Gol_3.mp4` | 100 | 1.7049 | 58.65 | 55.21 |
| `Video_Nao_Gol_4.mp4` | 100 | 1.6799 | 59.53 | 55.85 |

## Comparacao consolidada

| Backend | Clipes | Frames | Tempo medio por 100 frames (s) | FPS medio | Delta medio memoria (MB) |
|---|---:|---:|---:|---:|---:|
| CPU | 5 | 500 | 1.4951 | 73.39 | 17.57 |
| DirectML | 5 | 500 | 1.7086 | 58.55 | 55.65 |

## Conclusao

A comparacao numerica CPU vs DirectML foi executada com 5 clipes reais. Neste teste de fumaca, a CPU foi mais rapida e usou menos memoria que DirectML. Isso nao significa que a GPU sera pior no pipeline final, porque o teste atual faz apenas leitura, resize e operacoes simples de tensor; inferencia YOLO pode ter comportamento diferente.

- CPU validada com 5 clipes reais;
- tentativa CUDA executada e bloqueada por ambiente CPU-only;
- DirectML executado em ambiente separado;
- comparacao com RTX 3060 pendente para o computador da universidade.
