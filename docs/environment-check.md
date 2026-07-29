# Verificacao Inicial do Ambiente

Data: 19/07/2026

## Ferramentas de sistema

| Ferramenta | Status | Observacao |
|---|---|---|
| Git | Instalado | `git version 2.55.0.windows.3` |
| Python | Instalado | Python 3.11 em `C:\Users\mathe\AppData\Local\Programs\Python\Python311\python.exe` |
| FFmpeg | Instalado | `ffmpeg version 8.1.2-full_build-www.gyan.dev` |

Observacao: apos a instalacao via `winget`, a sessao atual do terminal ainda nao atualizou o PATH para `git`, `python` e `ffmpeg`. Os comandos funcionam usando os caminhos absolutos; uma nova sessao do terminal deve reconhecer os aliases globais.

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

## Observacoes

- A instalacao inicial do PyTorch encontrou o limite de caminho longo do Windows.
- Para concluir a instalacao sem alterar o registro do Windows, a pasta do projeto foi temporariamente mapeada como unidade `T:` com `subst`.
- O ambiente atual esta configurado para CPU, como previsto no escopo minimo do projeto.
