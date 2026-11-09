# Ambiente local de desenvolvimento e reprodução

**Inventário observado em 26/10/2026.** O desktop foi instalado e compilado em um snapshot temporário limpo. A receita Python/ROCm ainda não foi instalada numa cópia limpa.

## Ambiente Python/ROCm

| Componente | Versão observada | Papel |
|---|---|---|
| Python | 3.12.10 (`.venv-rocm`) | Versão escolhida para o ambiente Windows/ROCm e alinhada ao `pyproject.toml`. |
| PyTorch | 2.9.1+rocm7.2.1 | Runtime ROCm no Windows. |
| TorchVision | 0.24.1+rocm7.2.1 | Pareado com o PyTorch instalado. |
| OpenCV | 5.0.0.93 | `opencv-python`; o inventário local também detectou `opencv-python-headless`. |
| NumPy / pandas | 2.3.5 / 3.0.5 | Pins ajustados ao ambiente observado. |
| Ultralytics | 8.4.131 | Instalado localmente para scripts experimentais; opcional, fora da base do aplicativo. |
| Pillow / joblib / Pydantic | 12.3.0 / 1.5.3 / 2.13.5 | Dependências importadas diretamente pelo pipeline e API. |
| Node.js / npm | 24.19.0 / 11.17.0 | Desktop. |
| Electron / Vite / React | 44.9.3 / 8.3.5 / 19.3.0 | Versões diretas atualizadas após a auditoria npm. |
| FFmpeg | `ffmpeg-static` 5.3.0 | Executável presente em `desktop/node_modules/ffmpeg-static/ffmpeg.exe`; `ffmpeg` não está no `PATH`. O npm informou que o script postinstall não está aprovado. |

### Aplicativo em CPU (atualização de 05/11/2026)

O desenvolvimento do pacote Windows/CPU usa `.venv-cpu`, `requirements-cpu.txt` e
`scripts/build_cpu.ps1`, sem modificar o ambiente ROCm. Consulte
[build_cpu.md](build_cpu.md) para os comandos e limites de validação.
O aplicativo usa CPU por padrão; `GOAL_REVIEW_DEVICE=auto` ou `cuda:0` é uma opção
explícita no desenvolvimento. O pacote de teste usa CPU.

Correção do inventário de outubro: o aplicativo **importa Ultralytics indiretamente**
ao carregar `experiments/ball/mine_goal_event_candidates_v1.py`. Portanto, Ultralytics
é necessário no ambiente CPU, embora a importação não apareça diretamente em `app/`.

### Instalação Python

A receita histórica ROCm está em `requirements.txt`; ferramentas de desenvolvimento estão separadas em `requirements-dev.txt`, e dependências experimentais em `requirements-experiments.txt`. Essa receita ROCm ainda não foi instalada numa cópia limpa. `torchaudio` não é usado pelos scripts selecionados. A receita CPU inclui Ultralytics, necessário à mineração carregada dinamicamente pelo aplicativo V4.

Para PyTorch no Windows com ROCm 7.2.1, seguir a [instrução oficial da AMD para Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html), que fornece wheels para Python 3.12 e PyTorch 2.9.1/TorchVision 0.24.1. A página também informa os pré-requisitos de driver. Instalar primeiro os componentes/wheels ROCm e, depois, na raiz do repositório:

```powershell
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements.txt
```

Para ferramentas de desenvolvimento ou experimentos antigos, instalar conscientemente o grupo necessário:

```powershell
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-rocm\Scripts\python.exe -m pip install -r requirements-experiments.txt
```

Os arquivos opcionais preservam pins anteriormente declarados, mas ainda precisam de instalação e verificação de compatibilidade antes de serem tratados como reproduzíveis. O arquivo local ignorado `requirements-directml.txt` pertence a uma tentativa DirectML antiga e não é a receita do aplicativo V4.

## Instalação do desktop

`desktop/package.json` e `desktop/package-lock.json` agora registram as mesmas versões exatas das dependências diretas. No PowerShell, `npm.cmd` evita o bloqueio de `npm.ps1` pela política de execução. Para instalar a partir do lockfile, dentro de `desktop/`:

```powershell
npm.cmd ci
npm.cmd run build
```

`npm.cmd ci` foi concluído em 26/10/2026 fora do sandbox, inclusive após as atualizações de segurança: 104 pacotes instalados e `npm ls --depth=0` sem dependências ausentes. `npm audit fix` aplicou somente atualizações compatíveis e reportou zero vulnerabilidades. O npm avisou que o script postinstall de `ffmpeg-static` não está aprovado; o executável está presente nesta instalação. A primeira tentativa dentro do sandbox falhou por `EACCES` ao download HTTPS. `npm.cmd run build` passou com Vite 8.3.5. O arquivo de configuração foi renomeado para `.mjs` para remover o aviso CJS/ESM. O build Vite não empacota Python, pesos nem transforma o aplicativo em instalador autocontido.

## Limites e próximas verificações

- Os pins base refletem as versões observadas, mas a receita ainda não foi instalada numa cópia limpa.
- As dependências opcionais (`pytest`, Ruff, mypy, XGBoost e ONNX Runtime) não estavam instaladas no ambiente observado.
- Há um instalador CPU preliminar gerado e exercitado localmente em 05/11/2026;
  instalação e validação em outra máquina continuam pendentes. Evidências em `docs/build_cpu.md`.
- A aplicação operacional V4 foi exercitada em cenários manuais; isso não garante reprodução limpa ou desempenho em outro hardware.
- O artigo e sua compilação estão no Overleaf, fora do escopo destas instruções.
