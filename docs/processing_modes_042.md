# Pacote completo offline 0.4.2

O executável `dist/complete/GoalEventReview-0.4.2-offline-x64.exe` reúne o aplicativo e três ambientes Python independentes: CPU, NVIDIA/CUDA e AMD/ROCm. Não é necessário instalar Python nem baixar componentes na seleção do modo.

Na primeira abertura, uma janela verifica SHA-256 e prepara os arquivos em `%LOCALAPPDATA%\GoalEventReview\offline\app-<identificador>`. Nas aberturas seguintes, o aplicativo reutiliza essa preparação. É necessário espaço livre para o conteúdo descompactado além do próprio executável. O arquivo `.json` junto ao pacote registra os tamanhos medidos e a identidade do conteúdo.

## Seleção e compatibilidade

A abertura apresenta CPU, GPU NVIDIA e GPU AMD. Somente o ambiente escolhido é iniciado. A validação consulta os adaptadores Windows, confere o fornecedor do PyTorch e executa alocação, multiplicação de matrizes, convolução, NMS e inferência com os dois detectores incluídos. A janela permanece na seleção se a configuração falhar; não há troca silenciosa de GPU por CPU. O dispositivo aprovado é passado à API e registrado na sessão.

Os drivers de vídeo devem estar instalados no Windows. Incluir o runtime não torna todas as placas compatíveis, especialmente no caso AMD. A verificação de operações reais protege também contra placas detectadas cujo driver ou arquitetura não é aceito pelo runtime. Os registros ficam em `diagnostics` na pasta de dados do aplicativo.

No Windows, algumas bibliotecas nativas ROCm falham em caminhos longos. Para o processo AMD, o launcher cria uma junção em `%TEMP%\ger\<identificador>` apontando para o componente incluído. Isso não duplica os pacotes. O destino é conferido antes do uso. Os caches MIOpen ficam na pasta gravável do usuário.

## Dependências e reprodução

- CPU: PyTorch 2.9.1+cpu e torchvision 0.24.1+cpu.
- NVIDIA: PyTorch 2.9.1+cu126 e torchvision 0.24.1+cu126.
- AMD: PyTorch 2.9.1+rocm7.2.1, torchvision 0.24.1+rocm7.2.1 e os SDKs ROCm do ambiente isolado.
- Modelos, FFmpeg e dependências comuns são incluídos em cada componente conforme `packaging/backend_cpu.spec`.

Para reconstruir todos os ambientes já preparados e o pacote:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_complete.ps1 -RebuildEngines
```

Para empacotar componentes já construídos, omita `-RebuildEngines`. Os manifestos `component.json` precisam corresponder à versão do desktop. O empacotador usa 7-Zip/LZMA2 e um launcher Windows x64 baseado no .NET Framework presente no Windows 11. A compactação usa um núcleo, dicionário de 32 MiB e blocos de 64 MiB. Arquivos com SHA-256 idêntico são armazenados uma única vez no pacote e copiados aos destinos durante a preparação. O empacotador recusa um `.exe` que atinja 4 GiB, limite suportado pelo Windows. A preparação usa streaming para evitar carregar os runtimes inteiros na RAM e precisa de espaço temporário para o arquivo compactado além dos componentes extraídos.

O extrator standalone 7-Zip 26.06 é incluído como recurso do launcher; não precisa estar instalado no computador de destino. O build verifica seu SHA-256 fixado no código. `scripts/fetch_packaging_tools.py` obtém o binário e as licenças do repositório oficial para a máquina de desenvolvimento. As licenças LGPL e a referência ao código-fonte são gravadas junto ao aplicativo durante a preparação. Referências: [7-Zip](https://www.7-zip.org/), [código-fonte da versão incluída](https://github.com/ip7z/7zip/tree/26.06) e [limite de tamanho de executáveis descrito pelo Inno Setup](https://jrsoftware.org/files/is6-whatsnew.htm).

## Construção com pouca memória

Depois de preparar os componentes, execute apenas a compactação sob supervisão:

```powershell
.venv-cpu\Scripts\python.exe scripts/run_guarded.py --min-free-gb 3 --report build/guard/package_042.json -- .venv-cpu\Scripts\python.exe scripts/package_complete.py --source dist/complete/win-unpacked --output dist/complete/GoalEventReview-0.4.2-offline-x64.exe
```

O supervisor usa baixa prioridade, impede duas tarefas supervisionadas simultâneas e encerra somente a tarefa iniciada por ele se a RAM disponível cair abaixo do limite. Também recusa a continuação se detectar mais de 100 processos de busca `rg.exe`. O relatório registra o menor valor observado de memória disponível. Execute os testes de abertura e inferência separadamente, após o empacotamento. A supervisão não controla o consumo dos outros aplicativos do Windows.

## Validação local em 09/11/2026

- Executável: `dist/complete/GoalEventReview-0.4.2-offline-x64.exe`, 3.656.296.998 bytes (3,66 GB).
- Componentes descompactados: aproximadamente 11,68 GB. A primeira preparação precisa de cerca de 16 GB livres além do executável, para acomodar os arquivos temporários.
- Primeira preparação completa e abertura em CPU: aprovadas. A reabertura pelo cache e a seleção AMD também passaram.
- Seleção NVIDIA na máquina somente AMD: recusada com mensagem explícita, mantendo a janela de seleção aberta.
- Testes técnicos do componente CPU e do ambiente AMD extraído: análise de um trecho TRAIN de 90 segundos, clipe MP4, exportações CSV/JSON/HTML, anotação e restauração da sessão aprovados. Não são benchmarks comparativos.
- Supervisor: testes de recusa por pouca RAM e de encerramento de subprocessos após a saída do processo pai aprovados. Na retomada supervisionada, a menor RAM disponível observada foi 3,68 GiB; não houve acúmulo de buscas do editor.

O checksum está em `dist/complete/SHA256SUMS.txt`. O relatório `dist/complete/validation-0.4.2.json` reúne os resultados e os registros de memória. Este é um pacote local para teste, sem publicação automática.

## Limites de validação

A máquina de desenvolvimento tem uma AMD Radeon RX 9060 XT. A execução NVIDIA em uma placa NVIDIA precisa ser testada em outro computador; a rejeição quando a placa não existe é verificável localmente. A presença dos três componentes não equivale à validação de todas as GPUs ou versões de driver. Testes técnicos em TRAIN não modificam nem reabrem TEST V3 e não constituem uma nova avaliação científica.

Referências dos fornecedores: [versões PyTorch](https://pytorch.org/get-started/previous-versions/), [semântica HIP no PyTorch](https://docs.pytorch.org/docs/stable/notes/hip.html) e [compatibilidade AMD no Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityrad/windows/windows_compatibility.html).
