# Verificacao Inicial do Ambiente

Data: 29/07/2026

## Ferramentas de sistema

| Ferramenta | Status | Observacao |
|---|---|---|
| Git | Instalado | `git version 2.55.0.windows.3` |
| Python | Instalado | Python 3.11.9 no ambiente `.venv` |
| FFmpeg | Instalado e validado pelo usuario | `ffmpeg -version` retornou `ffmpeg version 8.1.2-full_build-www.gyan.dev` em novo PowerShell; a sessao do agente ainda nao recarregou o PATH |

Observacao: o `git` ja esta disponivel no PATH da sessao atual. O `ffmpeg` ainda precisa ser adicionado ao PATH ou referenciado por caminho absoluto.

## Ambiente Python

Ambiente virtual criado em `.venv`.

Comando de verificacao executado:

```powershell
.\.venv\Scripts\python.exe -c "import cv2, torch, ultralytics, pandas, sklearn, streamlit; print('OK'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

Resultado:

```text
OK
torch 2.13.0+cpu
cuda False
```

Verificacao complementar:

```powershell
.\.venv\Scripts\python.exe -c "import torchvision, torchaudio, onnx, onnxruntime; print('OK')"
```

Resultado:

```text
OK
```

Script de verificacao reproduzivel criado:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

Configuracoes adicionadas em `pyproject.toml`:

- `pytest` com descoberta em `tests`;
- `ruff` com regras iniciais de lint;
- `mypy` apontando para `app`, `scripts`, `src` e `tests`.

Validacoes executadas:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

Resultados:

```text
pytest: 2 passed
ruff: All checks passed
mypy: Success: no issues found in 3 source files
```

O script `scripts\check_environment.py` detectou corretamente que os pacotes Python obrigatorios estao disponiveis. Se ele for executado em uma sessao antiga que ainda nao recarregou o PATH, pode retornar falha para `ffmpeg`; em um novo PowerShell aberto na pasta do projeto, deve usar o PATH atualizado.

## Observacoes

- A instalacao inicial do PyTorch encontrou o limite de caminho longo do Windows.
- Para concluir a instalacao sem alterar o registro do Windows, a pasta do projeto foi temporariamente mapeada como unidade `T:` com `subst`.
- O ambiente atual esta configurado para CPU, como previsto no escopo minimo do projeto.
- A comparacao com GPU AMD sera registrada no spike de hardware quando houver backend acelerado definido e clipe de teste disponivel.
- Observacao operacional: para rodar comandos do projeto, abrir o terminal em `C:\Users\mathe\Documents\tcc_deteccao_analise_de_videos_de _futebol` ou executar `cd` para essa pasta antes de chamar `.\.venv\Scripts\python.exe`.
