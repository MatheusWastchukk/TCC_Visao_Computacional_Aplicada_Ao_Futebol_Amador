# Relatorio do Spike de Compatibilidade de Hardware

Data: 29/07/2026

## Objetivo

Decidir a estrategia inicial de execucao para desenvolvimento, inferencia e possivel treinamento do prototipo de deteccao de eventos em videos de futebol.

## Ambiente testado nesta etapa

| Item | Resultado |
|---|---|
| Sistema | Windows, executando no computador pessoal |
| Python | 3.11.9 no ambiente `.venv` |
| PyTorch | 2.13.0+cpu |
| CUDA | Indisponivel neste ambiente (`torch.cuda.is_available() == False`) |
| FFmpeg | Instalado e validado pelo usuario com `ffmpeg version 8.1.2-full_build-www.gyan.dev` |
| Videos de teste | Ainda nao ha clipes em `data/clips` para teste de 100 frames |

## Verificacoes executadas

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import cv2, torch, ultralytics, pandas, sklearn, streamlit; print('OK'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
winget list --id Gyan.FFmpeg --source winget --accept-source-agreements
```

Resultado observado:

```text
Python 3.11.9
OK
torch 2.13.0+cpu
cuda False
FFmpeg Gyan.FFmpeg 8.1.2
```

## Interpretacao

O ambiente atual atende ao nucleo minimo de portabilidade em CPU. A GPU AMD nao foi validada ainda porque a instalacao atual do PyTorch e CPU-only e o backend acelerado para a RX 9060 XT no Windows precisa ser decidido e documentado antes de uma comparacao justa.

O FFmpeg esta instalado, mas a sessao atual nao consegue chamar `ffmpeg` diretamente. Antes dos recortes de videos da TCC-011, o PATH deve ser atualizado ou o caminho absoluto do executavel deve ser registrado.

## Teste de fumaca em CPU

Como ainda nao ha clipes reais em `data/clips` ou `data/raw`, foi executado um teste de fumaca com video sintetico de 100 frames. Esse teste valida a rotina de leitura/processamento e o registro de metricas, mas nao substitui a comparacao final com videos reais.

Comando executado:

```powershell
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --device cpu --frames 100
```

Resultado salvo em `experiments/hardware-spike/cpu_smoke_test.json`:

| Metrica | Valor |
|---|---:|
| Frames solicitados | 100 |
| Frames processados | 100 |
| Tempo total | 0.4707 s |
| FPS medio | 212.46 |
| Memoria antes | 210.11 MB |
| Memoria depois | 221.00 MB |
| Delta de memoria | 10.89 MB |
| PyTorch | 2.13.0+cpu |
| CUDA disponivel | false |

## Teste CPU com clipes reais

Foram adicionados 5 clipes reais em `data/clips` e o mesmo teste de 100 frames foi executado em CPU para cada arquivo.

Comandos executados:

```powershell
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Gol_1.mp4 --device cpu --frames 100 --output experiments\hardware-spike\cpu_Video_Gol_1.json
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Nao_Gol_1.mp4 --device cpu --frames 100 --output experiments\hardware-spike\cpu_Video_Nao_Gol_1.json
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Nao_Gol_2.mp4 --device cpu --frames 100 --output experiments\hardware-spike\cpu_Video_Nao_Gol_2.json
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Nao_Gol_3.mp4 --device cpu --frames 100 --output experiments\hardware-spike\cpu_Video_Nao_Gol_3.json
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Nao_Gol_4.mp4 --device cpu --frames 100 --output experiments\hardware-spike\cpu_Video_Nao_Gol_4.json
```

| Clipe | Frames | Tempo total (s) | FPS | Delta memoria (MB) |
|---|---:|---:|---:|---:|
| `Video_Gol_1.mp4` | 100 | 1.6795 | 59.54 | 18.05 |
| `Video_Nao_Gol_1.mp4` | 100 | 1.6978 | 58.90 | 17.82 |
| `Video_Nao_Gol_2.mp4` | 100 | 1.6134 | 61.98 | 17.40 |
| `Video_Nao_Gol_3.mp4` | 100 | 1.7021 | 58.75 | 17.34 |
| `Video_Nao_Gol_4.mp4` | 100 | 0.7825 | 127.79 | 17.25 |

Resumo:

| Metrica | Valor |
|---|---:|
| Clipes reais testados | 5 |
| Frames reais processados | 500 |
| Tempo medio por 100 frames | 1.4951 s |
| FPS medio | 73.39 |
| Delta medio de memoria | 17.57 MB |

As consultas diretas de hardware via WMI/CIM retornaram acesso negado nesta sessao, portanto versao de driver e modelo exato da GPU devem ser registrados manualmente ou em uma sessao com permissao adequada.

## Protocolo pendente para teste controlado

Quando houver pelo menos um clipe curto em `data/clips`, executar exatamente os mesmos 100 frames em cada ambiente:

| Ambiente | Backend | Comando previsto | Metricas |
|---|---|---|---|
| Computador pessoal | CPU | `python -m app.cli analyze --video <clip> --frames 100 --device cpu` | tempo total, FPS, memoria |
| Computador pessoal | AMD, se suportado | comando equivalente com backend documentado | tempo total, FPS, memoria |
| Universidade | NVIDIA CUDA | comando equivalente com `--device cuda` | tempo total, FPS, memoria |

## Decisao inicial

| Decisao | Justificativa |
|---|---|
| Desenvolvimento principal em CPU | Ja funciona no ambiente atual e reduz risco de configuracao. |
| GPU AMD como otimizacao | Deve ser testada depois que o pipeline base estiver funcional. |
| RTX 3060 para treinamento, se necessario | CUDA tende a ter suporte mais direto para PyTorch e YOLO. |
| Treinamento personalizado fica condicionado aos resultados do baseline | Evita comprometer cronograma antes de medir a deteccao inicial. |

## Estrategia de contingencia

Se a RX 9060 XT nao oferecer caminho estavel no Windows, manter inferencia local em CPU para desenvolvimento, executar treinamentos na RTX 3060 da universidade e exportar pesos ou modelo ONNX apenas se isso simplificar a demonstracao.

## Status dos criterios de aceite

| Criterio | Status |
|---|---|
| Relatorio com CPU e GPU testados ou justificativa para GPU AMD | Parcial: CPU validada; GPU AMD justificada como pendente por backend CPU-only. |
| Tempo de inferencia registrado | Parcial: CPU com video sintetico e 5 clipes reais executada; falta comparacao GPU. |
| Estrategia principal definida | Concluido: CPU primeiro. |
| Estrategia de contingencia definida | Concluido: RTX 3060 para treinamento e CPU para portabilidade. |

## Tentativa de execucao em GPU

Foi tentada uma execucao com `--device cuda` no clipe `Video_Gol_1.mp4`.

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\hardware_smoke_test.py --video data\clips\Video_Gol_1.mp4 --device cuda --frames 100 --output experiments\hardware-spike\gpu_Video_Gol_1.json
```

Resultado:

```text
RuntimeError: Device 'cuda' requested, but CUDA is not available
```

Verificacoes do ambiente:

```text
torch 2.13.0+cpu
cuda_available False
cuda_device_count 0
torch_directml False
onnxruntime_providers ['AzureExecutionProvider', 'CPUExecutionProvider']
```

Evidencias adicionais:

- `experiments/hardware-spike/gpu_availability.json`;
- `experiments/hardware-spike/comparacao_cpu_gpu.md`.

## Execucao DirectML em GPU AMD

Para testar a GPU AMD sem alterar a `.venv` principal, foi criado o ambiente separado `.venv-dml` e registradas as dependencias em `requirements-directml.txt`.

Validacao:

```powershell
.\.venv-dml\Scripts\python.exe -c "import torch, torch_directml; print('torch', torch.__version__); print('dml_device', torch_directml.device()); print('device_count', torch_directml.device_count())"
```

Resultado:

```text
torch 2.4.1+cpu
dml_device privateuseone:0
device_count 1
```

Resultados com os mesmos 100 frames por clipe:

| Clipe | CPU FPS | DirectML FPS | CPU tempo (s) | DirectML tempo (s) | CPU memoria (MB) | DirectML memoria (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `Video_Gol_1.mp4` | 59.54 | 57.20 | 1.6795 | 1.7484 | 18.05 | 56.05 |
| `Video_Nao_Gol_1.mp4` | 58.90 | 57.43 | 1.6978 | 1.7413 | 17.82 | 55.75 |
| `Video_Nao_Gol_2.mp4` | 61.98 | 59.94 | 1.6134 | 1.6683 | 17.40 | 55.39 |
| `Video_Nao_Gol_3.mp4` | 58.75 | 58.65 | 1.7021 | 1.7049 | 17.34 | 55.21 |
| `Video_Nao_Gol_4.mp4` | 127.79 | 59.53 | 0.7825 | 1.6799 | 17.25 | 55.85 |

Resumo comparativo:

| Backend | Clipes | Frames | Tempo medio por 100 frames (s) | FPS medio | Delta medio memoria (MB) |
|---|---:|---:|---:|---:|---:|
| CPU | 5 | 500 | 1.4951 | 73.39 | 17.57 |
| DirectML | 5 | 500 | 1.7086 | 58.55 | 55.65 |

Conclusao parcial: neste teste leve, CPU foi mais rapida e usou menos memoria. A comparacao deve ser repetida no pipeline com inferencia YOLO, pois a GPU tende a ser mais relevante quando houver computacao de modelo, nao apenas leitura e preprocessamento.
