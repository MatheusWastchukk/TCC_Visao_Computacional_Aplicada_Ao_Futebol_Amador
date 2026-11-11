# Instalador Windows para testes — 0.4.2

O arquivo para enviar é `dist/installer/GoalEventReview-0.4.2-Setup-x64.exe`. Ele inclui CPU, NVIDIA e AMD, instala por usuário, cria atalhos e registra o desinstalador em Aplicativos instalados. Python, Node.js e modelos não são baixados no computador de destino. Os drivers de GPU compatíveis continuam necessários.

O instalador prepara os componentes durante a instalação. Os atalhos abrem diretamente o seletor de processamento; não há uma segunda preparação do pacote portátil. A opção de criar atalho na área de trabalho é facultativa. Sessões e anotações ficam separadas da pasta do programa e são preservadas na remoção.

O alvo local validado do aplicativo é Windows 11 x64. A execução em GPU NVIDIA ainda exige um teste positivo em outro computador com essa placa. A instalação completa ocupa aproximadamente 12 GB; reserve cerca de 16 GB livres para os componentes e temporários. O roteiro para os participantes está em [guia_teste_instalador.txt](guia_teste_instalador.txt), também incluído no instalador.

## Construção

Primeiro, prepare os runtimes e o arquivo completo seguindo [processing_modes_042.md](processing_modes_042.md). Mantenha a pasta `build/complete-bundle` e a aplicação em `dist/complete/win-unpacked`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build_installer.ps1 -Setup
```

O compilador oficial Inno Setup 7.1.1 é preparado em modo portátil, apenas em `build/tools/inno7`. O download é verificado por SHA-256 fixado no script. O binário usado na preparação inicial teve sua assinatura Authenticode validada, com editor Pyrsys B.V. O instalador gerado do aplicativo é um artefato de teste sem assinatura digital própria.

O build reutiliza o arquivo 7z validado, sem repetir a compressão dos modelos. Ele é armazenado internamente em blocos de 128 MiB para evitar o crescimento de RAM observado com uma única entrada de vários GB no compilador. Esses blocos ficam dentro de um único `.exe`; não precisam ser enviados separadamente.

Durante a instalação, os blocos são reunidos com buffer de 1 MiB, removidos, e o extrator standalone incluído descompacta os arquivos uma única vez no destino. As duplicatas são restauradas a partir dos arquivos canônicos já instalados. A receita gera uma lista explícita dos arquivos do pacote para o desinstalador; diretórios são removidos somente quando vazios. Não há exclusão recursiva da pasta inteira do aplicativo.

## Verificação isolada

Execute uma etapa por vez, usando o supervisor:

```powershell
.venv-cpu\Scripts\python.exe scripts/run_guarded.py --min-free-gb 3 --report build/guard/installer_install.json -- .venv-cpu\Scripts\python.exe scripts/smoke_installer.py install
.venv-cpu\Scripts\python.exe scripts/run_guarded.py --min-free-gb 3 --report build/guard/installer_integrity.json -- .venv-cpu\Scripts\python.exe scripts/smoke_installer.py verify
.venv-cpu\Scripts\python.exe scripts/run_guarded.py --min-free-gb 3 --report build/guard/installer_uninstall.json -- .venv-cpu\Scripts\python.exe scripts/smoke_installer.py uninstall
```

O teste usa `build/installer_e2e/app`, recusa substituir uma instalação já registrada e chama o desinstalador efetivamente registrado no Windows. A integridade compara SHA-256 dos arquivos instalados com a aplicação de origem. A remoção confere a preservação de um arquivo não instalado e de um marcador fora da pasta do programa.

O checksum de distribuição fica em `dist/installer/SHA256SUMS.txt`. Executáveis, blocos, arquivos `.iss` gerados, registros e cópias de teste permanecem em `build/` e `dist/`, ignorados pelo Git. Os fontes da receita ficam em `packaging/` e `scripts/`.

## Resultado local em 11/11/2026

O instalador desta revisão foi gerado e instalado em uma pasta isolada no Windows 11 x64. A instalação levou aproximadamente 178 segundos, e os 20.958 arquivos instalados coincidiram com a origem por SHA-256. O atalho apontou para o executável instalado; a abertura e a validação dos modos CPU e AMD passaram. A escolha de NVIDIA sem essa placa foi recusada com uma mensagem explicativa.

A desinstalação removeu os arquivos distribuídos e o registro do aplicativo, preservando o arquivo adicional de usuário e o marcador fora da instalação. Os dois testes do supervisor de memória também passaram. Os relatórios desta execução ficam em `dist/installer/validation.json` e `build/guard/installer_*.json`.

Estes testes verificaram instalação, integridade, seleção de hardware, abertura e remoção. O teste positivo em GPU NVIDIA e a instalação em um computador limpo continuam pendentes; a validação local não comprova compatibilidade com todas as placas AMD ou NVIDIA.

Referências: [Inno Setup — arquivos externos](https://jrsoftware.org/ishelp/topic_filessection.htm), [API de streams](https://jrsoftware.org/ishelp/topic_scriptclasses.htm) e [modo portátil do compilador](https://jrsoftware.org/ishelp/topic_technotes.htm).
