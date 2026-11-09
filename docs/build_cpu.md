# Pacote preliminar Windows/CPU

Esta rota prepara o aplicativo Electron + FastAPI V4 para computadores Windows x64
sem GPU dedicada e sem Python/Node.js instalados. Os modelos e parâmetros de
seleção continuam os mesmos. O pacote local é um teste de distribuição, não uma
nova avaliação de TEST V3 nem uma versão final liberada para redistribuição.

## Preparação e construção

Na raiz do projeto, com Python 3.12 e Node.js/npm disponíveis na máquina de build:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_cpu.ps1 -Setup -Python "C:\caminho\Python312\python.exe"
```

`-ExecutionPolicy Bypass` vale somente para esse processo e permite executar o
script local em máquinas cuja política padrão bloqueia `.ps1`; não altera a política
persistente do Windows.

O script cria `.venv-cpu`, instala PyTorch 2.9.1/TorchVision 0.24.1 pelo índice
oficial CPU, instala as dependências e ferramentas fixadas e obtém os binários
Electron/FFmpeg pelos scripts desses pacotes. `.venv-rocm` permanece independente.
O build exige os cinco modelos locais referenciados em `backend/server.py` e os
pesos ImageNet do ResNet18. Por padrão, procura estes últimos em
`%USERPROFILE%\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth`.
Para outra origem, informe `-EncoderWeights "C:\caminho\resnet18-f37072fd.pth"`.
O build falha se os recursos necessários não existirem; não substitui pesos.

Depois da preparação, para reconstruir após mudanças:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_cpu.ps1
```

Para produzir também o instalador NSIS:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_cpu.ps1 -Installer
```

Saídas:

- `build/cpu/backend/goal-review-api/`: backend PyInstaller em pasta.
- `dist/cpu/win-unpacked/Goal Event Review.exe`: aplicativo para teste local.
- `dist/cpu/Goal Event Review-0.4.1-cpu-x64.exe`: instalador atual, quando solicitado.
- `build/cpu/build_manifest.json`: inventário com SHA-256 dos arquivos e versões Python.

Para transferir a versão em pasta, copie **toda a pasta `win-unpacked`**. O `.exe`
sozinho depende dos arquivos ao redor. O instalador reúne esses componentes para
instalação. As saídas e o ambiente CPU são ignorados pelo Git.

## Comportamento da aplicação

A versão 0.4.1 corrige a amostragem de CPU e otimiza a recuperação de quadros da
mineração. A comparação técnica e os limites estão em
[cpu_diagnostico_20261108.md](cpu_diagnostico_20261108.md).

- CPU é o padrão também no desenvolvimento. Para manter GPU como opção, configure
  `$env:GOAL_REVIEW_DEVICE = 'auto'` ou `'cuda:0'` antes de iniciar o desktop.
  `auto` seleciona CPU quando não há GPU disponível; não promete recuperar uma
  falha de driver durante inferência. O pacote Electron força CPU.
- No desenvolvimento, o Electron procura `.venv-cpu`, `.venv-rocm` e `.venv`,
  nessa ordem. `GOAL_REVIEW_PYTHON` permite escolher outro interpretador.
- O backend empacotado roda de `resources/backend/goal-review-api.exe`.
- Os pesos ResNet18 são carregados do pacote, sem download automático. O caminho
  histórico de pesquisa continua disponível quando não se informa peso local.
- Sessões novas no Electron instalado ficam na subpasta `data/sessions` do
  diretório `userData` do Electron, sob `%APPDATA%` no Windows. O desenvolvimento preserva
  `outputs/review_app/sessions`. Não há migração automática das sessões antigas.
- Os vídeos originais ficam no local escolhido pelo usuário. Eles precisam continuar
  acessíveis para as funções que leem o vídeo original.
- FFmpeg acompanha o backend; não exige instalação no PATH.

## Verificação

O build executa `goal-review-api.exe --self-check` antes de montar o desktop.
Essa verificação importa a API, carrega os cinco modelos, executa os detectores em
uma imagem artificial, executa o ResNet18 local em CPU e chama o FFmpeg.
Ela não equivale à análise de uma partida nem à validação visual da interface.

Testes direcionados:

```powershell
.\.venv-cpu\Scripts\python.exe -m pytest tests/test_cpu_runtime.py tests/test_analyze_match_cli.py tests/test_generate_segment_clips.py tests/test_operational_modes_v4.py tests/test_temporal_inference_v4.py tests/test_visual_fusion_inference_v4.py tests/test_visual_encoder_local.py
npm.cmd run test:launcher --prefix desktop
```

Para exercitar o backend com um vídeo curto autorizado:

```powershell
.\.venv-cpu\Scripts\python.exe scripts/smoke_cpu_package.py --backend build/cpu/backend/goal-review-api/goal-review-api.exe --video C:/videos/trecho_autorizado.mp4 --workspace build/cpu/smoke_novo
```

O diretório de teste deve ser novo. O script usa a porta 8765 e recusa iniciar se
outra API já estiver nela. Verifica processamento em CPU, MP4, exportações, escrita
de nota e restauração após reiniciar o backend; grava relatório e logs na pasta indicada.

Antes de considerar a entrega pronta, testar em outro Windows sem Python/Node:
abrir o aplicativo, selecionar um vídeo autorizado, analisar, reproduzir os clipes,
anotar, exportar e fechar/reabrir o aplicativo para conferir as sessões. Medir tempo
e RAM no equipamento de destino. Desligar a rede nesse teste verifica a ausência
de dependências de download durante o uso.

## Limites de entrega

O pacote inclui pesos, Ultralytics e FFmpeg para os testes locais. A origem e as
obrigações de redistribuição desses componentes continuam sujeitas à auditoria
em [auditoria_licencas_pacote.md](auditoria_licencas_pacote.md). Não inclui vídeos,
clipes ou resultados TRAIN/TEST. Não é assinado digitalmente.

Referências técnicas: [PyInstaller em pasta](https://pyinstaller.org/en/stable/operating-mode.html),
[recursos do PyInstaller](https://www.pyinstaller.org/en/stable/runtime-information.html),
[NSIS/electron-builder](https://www.electron.build/v26/docs/nsis/).

## Verificação local em 05/11/2026

- Ambiente CPU instalado do zero em `.venv-cpu`: Python 3.12.10,
  PyTorch `2.9.1+cpu`, TorchVision `0.24.1+cpu`; `pip check` passou.
- Onze testes Python direcionados e dois testes Node do inicializador passaram.
- Build completo executado por `scripts/build_cpu.ps1 -Installer`, incluindo
  autoteste do backend, build Vite, empacotamento NSIS e manifesto.
- Instalador gerado: `dist/cpu/Goal Event Review-0.4.0-cpu-x64.exe`,
  434.538.417 bytes. A pasta `win-unpacked` teve 4.186 arquivos inventariados.
- O aplicativo empacotado abriu usando um perfil de teste isolado; a API respondeu,
  o React e os recursos locais carregaram e a captura da interface foi inspecionada.
  Evidências locais: `build/cpu/desktop_smoke/result.json` e `interface.png`.
- O backend da própria pasta de distribuição processou um trecho de 90 segundos
  de `match_04` (proveniência TRAIN), gerou um clipe MP4 H.264, respondeu às
  exportações CSV/JSON/HTML e preservou sessão e nota depois de reiniciar.
  Evidências: `build/cpu/smoke_match04_90s_packaged/smoke_result.json`,
  `backend.log` e os arquivos exportados na mesma pasta.
- O teste inicial do backend levou 44,727 s; o teste a partir do pacote levou
  124,079 s e ocorreu enquanto o instalador era gerado. São verificações funcionais,
  sem controle de carga para comparação de desempenho.
- A instalação do NSIS em outra máquina, reprodução visual dos clipes no player,
  teste sem rede e desempenho no computador de destino permanecem por validar.
  O teste de abertura da interface e o teste da API não cobrem todas as interações manuais.

Essas verificações não executaram inferência sobre TEST nem alteraram seus resultados
congelados. Nenhum vídeo de origem ou resultado de avaliação foi incluído no instalador.

## Atualização em 08/11/2026

O usuário relatou que o pacote anterior funcionou em outro computador, com CPU
sempre em zero no painel e previsão de 149 minutos para `match_32`. O indicador foi
corrigido na versão 0.4.1, que também inclui otimização da recuperação de quadros
e diagnóstico de tempos por etapa na exportação JSON. Os detalhes e a comparação
local estão em [cpu_diagnostico_20261108.md](cpu_diagnostico_20261108.md).
O usuário informou um Samsung Book Core i5-1135G7, 8 GB de RAM, SSD de 256 GB,
Iris Xe e Windows 11. Esclareceu que os 149 minutos eram uma previsão do aplicativo,
sem informar a duração final. Esse número não constitui benchmark do equipamento.
