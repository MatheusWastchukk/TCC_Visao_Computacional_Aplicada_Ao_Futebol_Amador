# Instalador com componentes GPU opcionais — 0.4.3

A versão online inclui aplicativo, modelos e processamento CPU. A instalação oferece CPU, NVIDIA ou AMD; somente o modo GPU escolhido baixa componentes adicionais. O instalador informa os fornecedores, domínios oficiais e tamanho aproximado antes do download. Os drivers de vídeo não são instalados nem alterados.

O catálogo versionado em `packaging/gpu_catalog.json` fixa Python 3.12/Windows x64, URLs HTTPS, versões, tamanhos e SHA-256. NVIDIA usa PyTorch 2.9.1/CUDA 12.8 e torchvision 0.24.1 do índice oficial PyTorch. AMD usa as versões ROCm 7.2.1 indicadas na documentação oficial AMD. O catálogo atual corresponde a aproximadamente 2,87 GB de downloads NVIDIA ou 2,19 GB AMD. Esses downloads não fazem parte do tamanho do instalador base.

## Correção NVIDIA em código — 19/11/2026

Um computador com RTX 5060 Ti apresentou `no kernel image is available for execution on the device` com o componente anterior CUDA 12.6. A [tabela oficial NVIDIA](https://developer.nvidia.com/cuda/gpus) identifica essa placa como Blackwell, capacidade 12.0. O catálogo e a receita NVIDIA passaram a usar CUDA 12.8, disponível para Windows com as mesmas versões de [PyTorch e torchvision](https://pytorch.org/get-started/previous-versions/#v291). Os tamanhos foram consultados por HEAD e os hashes obtidos do índice oficial; os wheels não foram baixados nesta alteração. O catálogo AMD permaneceu idêntico.

A seleção NVIDIA agora recusa antecipadamente uma GPU de capacidade 12.0 quando o componente carregado usa CUDA anterior a 12.8, com orientação para atualizar o aplicativo e reinstalar o componente NVIDIA. Com o componente novo, os testes reais de operações e modelos continuam obrigatórios antes da abertura.

**Estado em 19/11: alteração de código, sem novo build naquela etapa.** Os 20 testes leves de seleção e instalação de componentes passaram, incluindo a recusa de Blackwell com CUDA 12.6 e a continuidade da validação com CUDA 12.8 ou superior. O instalador anterior de 404.586.839 bytes contém o catálogo CUDA 12.6. A reconstrução está registrada abaixo; não há validação positiva em RTX 5060 Ti nesta revisão.

## Instalador reconstruído em 20/11/2026

O novo artefato `dist/online-installer/GoalEventReview-0.4.3-Online-Setup-x64.exe` tem **404.628.895 bytes (404,6 MB)** e SHA-256 `75bf3a34019368b098d36b3b4e6d6b91a6c8c4ddaa089ee45f6414adc441defb`. Inclui o backend atualizado, a coleta WDDM, as métricas recolhidas nos detalhes e o catálogo NVIDIA CUDA 12.8. CPU continua offline; os downloads opcionais são 2.871.663.543 bytes NVIDIA e 2.190.543.312 bytes AMD.

A reconstrução reutilizou as dependências CPU congeladas e recompilou os seis módulos alterados, conferidos contra os fontes no executável final. A interface foi compilada novamente e os 14 arquivos distribuídos da interface/launcher foram comparados com o ASAR. As etapas pesadas foram sequenciais e supervisionadas com limite mínimo de 3 GiB de RAM. O Electron já instalado no projeto foi reutilizado, sem download. Os pacotes GPU não foram baixados novamente.

Passaram 26 testes Python, 13 testes Node e a validação real do backend CPU empacotado com operações e os dois detectores. O relatório desta revisão fica em `dist/online-installer/validation.json`, separado das evidências históricas. A revisão anterior do relatório foi preservada em `build/release_20261120/validation_previous.json`.

A cópia empacotada abriu em CPU com perfil isolado e uma sessão fictícia, sem processar vídeo. A checagem confirmou métricas ocultas inicialmente e visíveis após expandir os detalhes; a captura de tela também foi inspecionada. NVIDIA ausente foi recusada antes da abertura. A pasta gerada pelo ambiente de build precisou receber leitura/execução para `ALL APPLICATION PACKAGES` para permitir o sandbox do Electron; nenhuma proteção foi desativada e a instalação do usuário não foi alterada.

Para atualizar uma instalação que usa o pacote NVIDIA anterior, escolha NVIDIA novamente no instalador: isso baixa e prepara o componente CUDA 12.8. Instalar apenas CPU atualiza a base, mas preserva o componente NVIDIA antigo. A GTX 1650 (Turing) e a RTX 5060 Ti (Blackwell) estão nas arquiteturas contempladas pelo componente escolhido; abertura e processamento nessas placas ainda precisam de teste real com drivers compatíveis. A instalação/desinstalação completa e o teste AMD desta reconstrução também não foram repetidos; os resultados anteriores permanecem históricos.

CPU funciona sem internet. Falha, cancelamento ou corrupção no download impede a preparação do componente GPU; o usuário pode tentar novamente ou voltar e escolher CPU. Antes do download, o instalador consulta os adaptadores Windows e recusa um fornecedor ausente. A abertura do aplicativo mantém a validação real de operações e modelos na GPU; detectar a marca da placa não comprova suporte do driver ou da arquitetura.

## Construção

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build_online_installer.ps1 -RebuildBase
```

Use `-Setup` também se for necessário preparar as dependências da base CPU. Com `dist/cpu/win-unpacked` já construído e correspondente à versão atual, omita `-RebuildBase`. A preparação do compilador reutiliza a receita existente do Inno Setup. Os builds são supervisionados com limite mínimo de 3 GiB de RAM disponível, sem tarefas pesadas simultâneas.

A saída é `dist/online-installer/GoalEventReview-0.4.3-Online-Setup-x64.exe`, acompanhada de JSON, SHA-256 e `LEIA-ME-TESTE.txt`. A revisão com caminho físico AMD de 16/11 tinha **404.586.839 bytes (404,6 MB)**; a reconstrução atual está descrita acima. O construtor recusa classificar o pacote como concluído se atingir 500.000.000 bytes. O instalador completo offline 0.4.2 permanece separado.

## Instalação e remoção dos componentes

O instalador baixa os arquivos para sua pasta temporária, com progresso e verificação SHA-256 pelo Inno Setup. O backend repete a verificação antes de extrair. Não há execução de pip, setup.py ou scripts recebidos pela rede. Os wheels e o módulo Python puro ROCm são extraídos com buffer limitado; caminhos externos e links são recusados.

Cada modo usa uma pasta isolada, publicada somente após a extração completa. Os componentes GPU compartilham o backend e os modelos CPU, mas carregam seu próprio PyTorch/torchvision antes dos módulos CPU congelados. CPU continua usando as bibliotecas incluídas. O backend obtém o nome curto físico NTFS do componente GPU com `GetShortPathNameW`, sem criar uma junção temporária nesse fluxo online.

O instalador e o desinstalador capturam a saída do preparador no log do Inno Setup. O desinstalador chama o backend antes de remover seus arquivos. Apenas arquivos registrados dos componentes opcionais são removidos, com diretórios excluídos somente se vazios. Sessões, anotações e arquivos adicionais não são apagados recursivamente. A variante online usa um identificador e diretório próprios para os testes, preservando os arquivos da instalação offline anterior.

Instalação silenciosa: `/GPU=cpu` (padrão), `/GPU=nvidia` ou `/GPU=amd`, além das opções usuais do Inno Setup. Selecionar GPU continua sujeito à verificação de hardware e aos downloads oficiais.

## Validação inicial em 16/11/2026

O instalador final foi instalado em uma pasta isolada no Windows 11 x64. A instalação CPU sem downloads adicionais e a abertura da interface passaram. A instalação AMD baixou e verificou os seis arquivos oficiais, publicou o componente e abriu o aplicativo usando a Radeon RX 9060 XT local. A escolha de NVIDIA ausente foi recusada tanto pelo instalador final quanto pelo aplicativo.

O mesmo trecho TRAIN autorizado de 90 segundos foi analisado em AMD e CPU após a instalação dos componentes GPU. Nos dois modos, a análise gerou clipe H.264, exportações CSV/JSON/HTML e restaurou a sessão com as anotações. É um smoke técnico, não um benchmark comparativo ou uma nova avaliação científica. TEST V3 não foi consultado.

A comparação SHA-256 dos 4.186 arquivos da base CPU instalada passou, assim como a comparação do catálogo. A desinstalação removeu os 17.939 arquivos GPU registrados e os arquivos distribuídos da base, preservando um arquivo adicional dentro do componente, outro na pasta do programa e um marcador externo. Os 21 testes Python e oito testes Node também passaram.

O backend foi reconstruído reutilizando o arquivo de módulos já analisado e lendo um módulo por vez para limitar a RAM; a receita limpa continua em `packaging/backend_cpu.spec`. Tentativas iniciais com pouca memória foram interrompidas pelo supervisor, sem reduzir o limite de 3 GiB. Após liberar memória, o empacotamento e os testes foram concluídos sequencialmente. Os resultados ficam em `dist/online-installer/validation.json`, `build/guard/online_*.json` e `build/online_installer_e2e`.

O teste positivo NVIDIA e a instalação desta revisão em um computador limpo continuam pendentes. A validação AMD local não comprova suporte de todas as placas ou drivers. Este é um instalador para testes, sem assinatura digital própria ou publicação automática.

## Primeira correção de caminhos — validação limitada

Após a instalação padrão, foi relatado `UNKNOWN: unknown error, stat` no alias `%TEMP%\ger\7ef6ffcee843`. A inspeção encontrou a junção apontando para o componente AMD instalado e os arquivos presentes. A causa de baixo nível não foi reproduzida de forma determinística; a validação antiga seguia a junção por `fs.realpathSync`, justamente a operação citada no erro.

A correção usa `original-fs` no Electron, lê o destino da junção sem segui-la e compara os destinos pela API nativa `realpathSync.native`. Não remove nem recria pastas ocupadas por arquivos ou destinos diferentes. Três testes de regressão cobrem a falha simulada `UNKNOWN`, a reutilização da junção e a preservação de caminhos ocupados; os 11 testes Node passaram.

A atualização local substituiu somente `Goal Event Review.exe` e `resources/app.asar`, com cópia de segurança em `build/repairs`. O backend, os modelos, o manifesto e os pacotes GPU já baixados foram preservados. O teste abriu em AMD o aplicativo instalado em `%LOCALAPPDATA%\Programs\Goal Event Review Online`, reutilizando a mesma junção reportada, com perfil de teste isolado e sem downloads. A validação de hardware e modelos passou, e a API informou `processing_backend=amd` e versão 0.4.3.

O instalador foi reconstruído com esse mesmo pacote de interface, cujo conteúdo ASAR foi comparado com o fonte. O SHA-256 desta revisão é `c86e4106e0fff0daf3d50db0097ac1a82e952626e2735651afe013949d3eed9f`. A instalação completa com downloads e as análises de vídeo descritas acima pertencem à validação inicial; nesta correção, os novos testes cobrem a resolução de caminhos, o pacote de interface e a abertura AMD na instalação real. Os relatórios iniciais foram preservados em `validation-before-amd-path.json`, e o relatório atual identifica a correção `amd-junction-native-v1`.

Uma tentativa com pouca RAM foi bloqueada antes de iniciar o empacotamento. Após liberar memória e fechar o aplicativo, a atualização local, o teste AMD e a geração do instalador passaram sob supervisão, mantendo o limite de 3 GiB. A versão permanece 0.4.3 para preservar o contrato com os componentes já instalados. Para atualizar outra instalação 0.4.3, feche o aplicativo e execute o instalador atualizado escolhendo CPU: essa opção atualiza a base e preserva o componente AMD existente, que pode ser escolhido na abertura do aplicativo.

## Correção validada para RedirectionGuard

Uma tentativa posterior registrou `WinError 448` no backend antes de gerar o JSON da verificação. A inspeção confirmou `EnforceRedirectionTrust=1` nos processos do aplicativo. Um processo de diagnóstico próprio com essa proteção ativa reproduziu a recusa da junção, enquanto o caminho físico curto NTFS foi acessível. A documentação do Inno Setup confirma que aplicativos abertos pelo instalador podem herdar essa proteção, inclusive com `PrivilegesRequired=lowest`.

A correção anterior foi testada por um launcher separado, sem essa proteção herdada; portanto, não cobriu a condição que falhou na abertura após a instalação. A nova implementação passa o diretório físico do componente ao backend e usa `GetShortPathNameW` para obter seu nome curto NTFS, sem criar uma junção nesse fluxo. O instalador deixa de abrir o aplicativo automaticamente e orienta a abertura pelo menu Iniciar. Nenhuma política de segurança do Windows é desativada. Em volumes sem nomes 8.3, a API pode devolver o caminho completo; a verificação real da GPU continua obrigatória.

O diagnóstico de um backend que encerra antes de produzir JSON agora distingue o bloqueio 448 de um possível problema de driver e informa o arquivo de log. Passaram 20 testes Python e 13 testes Node, incluindo um teste Python que habilita RedirectionGuard em um subprocesso e verifica o carregador pelo caminho físico.

A instalação anterior já havia sido removida quando o reparo foi retomado. A base corrigida foi reinstalada no caminho original e os seis componentes AMD foram restaurados do cache local verificado, sem novos downloads. O teste abriu o aplicativo instalado com perfil isolado e aprovou alocação, multiplicação, convolução, NMS e os dois detectores. A API informou versão 0.4.3 e `processing_backend=amd`.

O observador de processos confirmou `EnforceRedirectionTrust=1` tanto na interface quanto nos processos de verificação e API. Esse teste reproduz a condição de proteção que faltava na validação anterior. Nenhuma proteção do Windows foi desativada. Os registros ficam em `build/amd_protected_result.json`, `build/launcher_043_local-protected_amd` e `build/guard/amd_physical_*.json`.

O instalador atual foi reconstruído com o novo backend e a nova interface. Seu SHA-256 é `40ebc53dd2d0d33f53bd0c2e28230016fc38158f7d778511edd4cf0137f45b21`. A análise complementar do trecho TRAIN sob essa proteção foi bloqueada antes de iniciar porque a RAM disponível caiu para cerca de 1,2 GiB; não é apresentada como concluída. As análises de vídeo da validação inicial permanecem como evidência separada. A abertura AMD e os testes de operações/modelos sob RedirectionGuard passaram antes dessa queda, com o supervisor mantendo o limite de 3 GiB.

Fontes: [instalação oficial AMD/ROCm 7.2.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html), [versões oficiais PyTorch](https://pytorch.org/get-started/previous-versions/), [downloads com verificação SHA-256 no Inno Setup](https://jrsoftware.org/inno-setup/documentation/topic_isxfunc_downloadtemporaryfile.htm), [APIs originais de arquivos no Electron](https://www.electronjs.org/docs/latest/tutorial/asar-archives), [herança do RedirectionGuard no Inno Setup](https://jrsoftware.org/inno-setup/documentation/topic_setup_redirectionguard.htm), [GetShortPathNameW](https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getshortpathnamew).
