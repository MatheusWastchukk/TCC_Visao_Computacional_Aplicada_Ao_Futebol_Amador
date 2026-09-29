# Configuração da GPU AMD

## Política de execução do projeto

Usar a GPU AMD como primeira opção nos próximos experimentos sempre que o
backend e os operadores necessários forem compatíveis. Para PyTorch em GPU,
começar pelo ambiente `.venv-rocm`; para inferência ONNX compatível, considerar
`.venv-dml`. Fazer primeiro um smoke curto do fluxo real do experimento. Se a
GPU não for reconhecida, um operador falhar ou o resultado não puder ser
validado, registrar o erro e repetir em CPU (`.venv`). Toda comparação deve
registrar o ambiente e o dispositivo efetivamente usados. A preferência por
GPU não altera retroativamente os experimentos já executados em CPU.

Ferramentas de anotação e reprodução de vídeo não precisam de aceleração GPU;
essa política se aplica aos experimentos de inferência, extração e treinamento
que consigam usar o backend.

## Estado

A GPU identificada neste computador é uma AMD Radeon RX 9060 XT, com 16 GB de
memória dedicada.

Foi mantido o `.venv` original com PyTorch CPU para preservar a
reprodutibilidade dos treinamentos já executados. O ambiente separado
`.venv-dml` contém `torch-directml`, Ultralytics e ONNX Runtime DirectML.

O teste PyTorch-DirectML executa operações simples, mas o pipeline Ultralytics
falha no NMS com `Cannot set version_counter for inference tensor` e, após o
contorno de inferência, com erro interno durante a operação. Por isso, esse
caminho não deve ser usado para treinamento sem uma validação adicional.

O caminho ONNX foi validado com `DmlExecutionProvider`, usando o modelo
exportado `best.onnx`. Ele é adequado para testar inferência acelerada pela
GPU AMD, mas não substitui o treinamento PyTorch.

Também foi executado um smoke test de treinamento com o adaptador DirectML e
AMP desativado. O Ultralytics iniciou, mas o primeiro batch falhou porque o
operador `unique(..., return_counts=True)` não é implementado pelo DirectML.
Assim, o treinamento GPU permanece bloqueado nesta combinação de versões.

Como rota de treinamento, está sendo preparado um ambiente ROCm separado. A
matriz da AMD lista a RX 9060 XT entre as GPUs Radeon compatíveis no Windows,
mas a instalação atual exige Python 3.12 e driver compatível; a confirmação
final depende da conclusão desse ambiente.

## Atualização ROCm/PyTorch

O Python 3.12 foi instalado e o ambiente separado `.venv-rocm` foi criado com
ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1`, torchvision e Ultralytics. O PyTorch
reconhece a GPU como `cuda:0` e o teste matricial e uma convolução simples
executaram na RX 9060 XT.

O primeiro smoke test do treinamento Ultralytics, porém, falhou no primeiro
batch ao compilar o kernel de BatchNorm do MIOpen. O erro foi `type_traits file
not found`, seguido de `miopenStatusUnknownError`. A causa identificada é a
ausência de headers/ferramentas C++ do MSVC no ambiente (`cl.exe` e
`type_traits` não foram encontrados). Portanto, o treinamento via GPU ainda
não está validado.

Antes de repetir o treino, instalar o Visual Studio Build Tools com o workload
de desenvolvimento C++ para desktop. Em cada sessão PowerShell, usar também
um cache gravável do MIOpen:

```powershell
New-Item -ItemType Directory -Force -Path .rocm_cache\miopen_db | Out-Null
New-Item -ItemType Directory -Force -Path .rocm_cache\miopen_kernel | Out-Null
$env:MIOPEN_USER_DB_PATH = (Join-Path (Get-Location) '.rocm_cache\miopen_db')
$env:MIOPEN_SYSTEM_DB_PATH = $env:MIOPEN_USER_DB_PATH
$env:MIOPEN_CUSTOM_CACHE_DIR = (Join-Path (Get-Location) '.rocm_cache\miopen_kernel')
```

Depois que o Build Tools estiver instalado, repetir o smoke test de uma época
com `--device 0`. Somente se ele completar deve-se iniciar o experimento longo.

O Visual Studio Build Tools 2022 foi instalado com o workload C++. O smoke test
foi repetido com o cache configurado e `--no-amp`: completou 1 época em cerca
de 145 segundos, sem `NaN`, usando `cuda:0` na RX 9060 XT. Os mAPs zerados desse
teste não são resultado científico; servem apenas para confirmar que o
treinamento Ultralytics funciona na GPU.

Comando-base validado para os próximos experimentos:

```powershell
.\.venv-rocm\Scripts\python.exe experiments\ball\run_ball_exp0001.py `
  --data experiments\ball\ball_dataset_v2.yaml `
  --model yolo11n.pt `
  --output experiments\ball\BALL-EXP-0004 `
  --epochs 30 --imgsz 640 --batch 8 --device 0 --no-amp
```

O primeiro experimento real via ROCm foi concluído como `BALL-EXP-0004` em
1355 segundos (aproximadamente 22,6 minutos), sem NaN. Com 30 épocas, 640 px e
batch 8, obteve precisão de 45,91%, recall de 26,75%, mAP50 de 25,02% e
mAP50-95 de 7,12%. Esses valores ainda precisam da auditoria visual manual
para comparação justa com os experimentos anteriores.

A auditoria visual foi gerada para as 119 imagens de validação. A classificação
automática inicial indicou 14 verdadeiros positivos, 59 falsos negativos e 46
verdadeiros negativos; esses números são apenas uma triagem e aguardam a
revisão manual.

A revisão manual foi concluída: 65 corretos, 54 falsos negativos, 0 falsos
positivos e 0 inconclusivos. Considerando apenas os 73 frames localizáveis
(VISIBLE/PARTIAL), o recall manual foi 19/73 = 26,03%, igual ao recall
automático registrado na validação do experimento.

## Comparação dos experimentos da bola

| Experimento | Ambiente | Épocas | Precisão | Recall | mAP50 | mAP50-95 | Recall manual |
|---|---|---:|---:|---:|---:|---:|---:|
| BALL-EXP-0001 | CPU | 10 | 24,60% | 15,10% | 11,60% | 4,42% | 13,70% (10/73) |
| BALL-EXP-0002 | CPU | 30 | 29,26% | 28,77% | 21,19% | 7,08% | 21,92% (16/73) |
| BALL-EXP-0003 | CPU | 30 | **55,96%** | 26,03% | **27,25%** | **7,85%** | **28,77% (21/73)** |
| BALL-EXP-0004 | ROCm/GPU | 30 | 45,91% | **26,75%** | 25,02% | 7,12% | 26,03% (19/73) |

O `BALL-EXP-0003` permanece como melhor baseline de qualidade. O
`BALL-EXP-0004` não superou esse modelo, mas terminou em 1355 segundos,
contra 5305 segundos do `BALL-EXP-0003`, aproximadamente 3,9 vezes mais
rápido. A GPU está validada como infraestrutura de treinamento, mas ainda
não como melhoria de qualidade.

Para a próxima etapa, deve-se manter o `BALL-EXP-0003` como referência,
usar ROCm/GPU e testar melhorias para objetos pequenos, como resolução e
amostragem, antes de modificar a arquitetura.

Com a auditoria do `BALL-EXP-0005` concluída, o recall manual em frames
localizáveis foi 41,10% (30/73), acima dos 28,77% (21/73) do `BALL-EXP-0003`.
Também foram registradas 12 caixas manuais para casos em que a revisão visual
identificou uma posição da bola não representada adequadamente na anotação
original. Essas caixas não alteram retroativamente as métricas do detector.

## Smoke ROCm para auditoria de propostas BALL — 25/09/2026

No ambiente atual, `torch.cuda.is_available()` e a inferência real dos pesos do
minerador funcionaram em `cuda:0`, identificando AMD Radeon RX 9060 XT e PyTorch
`2.9.1+rocm7.2.1`. O smoke de três frames em `match_06` terminou; o primeiro
frame teve custo de compilação/aquecimento, e os dois seguintes completaram em
0,208 s e 0,035 s (detector GOAL + BALL). Uma mensagem de parsing de
`offload-arch.exe` acompanhou a inicialização, mas não bloqueou esse smoke.

O fluxo da auditoria de propostas também pontuou 400 frames TRAIN completos em
ROCm, usando o mesmo backend, dispositivos e pesos congelados. O resultado foi
registrado em `experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/run_v2/inference_runtime_v1.json`.
Isso valida esta inferência frame a frame; não certifica treinamento ou todos os
fluxos ROCm. O minerador de partidas completas permaneceu sem progresso nas
tentativas anteriores. Para repetir experimentos, manter smoke do fluxo real,
cache MIOpen gravável e monitoramento de progresso antes de escalar.

## Cache MIOpen local para execução no workspace — 26/09/2026

Na ablação de resolução BALL, o ROCm reconheceu a RX 9060 XT, mas MIOpen falhou
ao abrir um lock em `C:\Users\mathe\.miopen\db\miopen-lockfiles` com acesso
negado. A ACL do caminho dava leitura ao grupo do sandbox e controle total à
conta proprietária. Redirecionar o User DB e o cache de kernels para diretórios
graváveis do workspace permitiu completar tanto um smoke como o sweep ROCm de
400 frames em três resoluções. Essa é a causa mais provável, não uma prova de
que o ACL era o único fator.

Os scripts da auditoria BALL configuram estes padrões antes de importar PyTorch:

- `MIOPEN_USER_DB_PATH`: `experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1/user_db`
- `MIOPEN_CUSTOM_CACHE_DIR`: `experiments/ball/GOAL-EVENT-BALL-PROPOSAL-QUALITY-AUDIT-EXP-0001/miopen_workspace_cache_v1/kernel_cache`

Variáveis já definidas pelo usuário são preservadas. A documentação oficial
MIOpen permite definir as rotas do User DB em runtime e usa
`MIOPEN_CUSTOM_CACHE_DIR` para o cache de kernels: [AMD ROCm MIOpen cache e
performance database](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/cache.html).

## Teste

```powershell
.\.venv-dml\Scripts\python.exe experiments\ball\directml_smoke_test.py `
  --model runs\detect\experiments\ball\BALL-EXP-0002\weights\best.onnx `
  --image experiments\ball\data\yolo_v2\images\validation\match_03_ball_001.jpg
```

Resultado esperado: `provedores` contendo `DmlExecutionProvider` e `status=ok`.
