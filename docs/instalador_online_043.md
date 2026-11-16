# Instalador com componentes GPU opcionais — 0.4.3

A versão online inclui aplicativo, modelos e processamento CPU. A instalação oferece CPU, NVIDIA ou AMD; somente o modo GPU escolhido baixa componentes adicionais. O instalador informa os fornecedores, domínios oficiais e tamanho aproximado antes do download. Os drivers de vídeo não são instalados nem alterados.

O catálogo versionado em `packaging/gpu_catalog.json` fixa Python 3.12/Windows x64, URLs HTTPS, versões, tamanhos e SHA-256. NVIDIA usa PyTorch 2.9.1/CUDA 12.6 e torchvision 0.24.1 do índice oficial PyTorch. AMD usa as versões ROCm 7.2.1 indicadas na documentação oficial AMD. O catálogo atual corresponde a aproximadamente 2,59 GB de downloads NVIDIA ou 2,19 GB AMD. Esses downloads não fazem parte do tamanho do instalador base.

CPU funciona sem internet. Falha, cancelamento ou corrupção no download impede a preparação do componente GPU; o usuário pode tentar novamente ou voltar e escolher CPU. Antes do download, o instalador consulta os adaptadores Windows e recusa um fornecedor ausente. A abertura do aplicativo mantém a validação real de operações e modelos na GPU; detectar a marca da placa não comprova suporte do driver ou da arquitetura.

## Construção

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build_online_installer.ps1 -RebuildBase
```

Use `-Setup` também se for necessário preparar as dependências da base CPU. Com `dist/cpu/win-unpacked` já construído e correspondente à versão atual, omita `-RebuildBase`. A preparação do compilador reutiliza a receita existente do Inno Setup. Os builds são supervisionados com limite mínimo de 3 GiB de RAM disponível, sem tarefas pesadas simultâneas.

A saída é `dist/online-installer/GoalEventReview-0.4.3-Online-Setup-x64.exe`, acompanhada de JSON, SHA-256 e `LEIA-ME-TESTE.txt`. A revisão validada tem **404.547.856 bytes (404,5 MB)**, abaixo da meta de 500 MB. O construtor recusa classificar o pacote como concluído se atingir 500.000.000 bytes. O instalador completo offline 0.4.2 permanece separado.

## Instalação e remoção dos componentes

O instalador baixa os arquivos para sua pasta temporária, com progresso e verificação SHA-256 pelo Inno Setup. O backend repete a verificação antes de extrair. Não há execução de pip, setup.py ou scripts recebidos pela rede. Os wheels e o módulo Python puro ROCm são extraídos com buffer limitado; caminhos externos e links são recusados.

Cada modo usa uma pasta isolada, publicada somente após a extração completa. Os componentes GPU compartilham o backend e os modelos CPU, mas carregam seu próprio PyTorch/torchvision antes dos módulos CPU congelados. CPU continua usando as bibliotecas incluídas. No AMD, uma junção temporária oferece um caminho curto às bibliotecas nativas, como no pacote offline.

O instalador e o desinstalador capturam a saída do preparador no log do Inno Setup. O desinstalador chama o backend antes de remover seus arquivos. Apenas arquivos registrados dos componentes opcionais são removidos, com diretórios excluídos somente se vazios. Sessões, anotações e arquivos adicionais não são apagados recursivamente. A variante online usa um identificador e diretório próprios para os testes, preservando os arquivos da instalação offline anterior.

Instalação silenciosa: `/GPU=cpu` (padrão), `/GPU=nvidia` ou `/GPU=amd`, além das opções usuais do Inno Setup. Selecionar GPU continua sujeito à verificação de hardware e aos downloads oficiais.

## Validação local em 16/11/2026

O instalador final foi instalado em uma pasta isolada no Windows 11 x64. A instalação CPU sem downloads adicionais e a abertura da interface passaram. A instalação AMD baixou e verificou os seis arquivos oficiais, publicou o componente e abriu o aplicativo usando a Radeon RX 9060 XT local. A escolha de NVIDIA ausente foi recusada tanto pelo instalador final quanto pelo aplicativo.

O mesmo trecho TRAIN autorizado de 90 segundos foi analisado em AMD e CPU após a instalação dos componentes GPU. Nos dois modos, a análise gerou clipe H.264, exportações CSV/JSON/HTML e restaurou a sessão com as anotações. É um smoke técnico, não um benchmark comparativo ou uma nova avaliação científica. TEST V3 não foi consultado.

A comparação SHA-256 dos 4.186 arquivos da base CPU instalada passou, assim como a comparação do catálogo. A desinstalação removeu os 17.939 arquivos GPU registrados e os arquivos distribuídos da base, preservando um arquivo adicional dentro do componente, outro na pasta do programa e um marcador externo. Os 21 testes Python e oito testes Node também passaram.

O backend foi reconstruído reutilizando o arquivo de módulos já analisado e lendo um módulo por vez para limitar a RAM; a receita limpa continua em `packaging/backend_cpu.spec`. Tentativas iniciais com pouca memória foram interrompidas pelo supervisor, sem reduzir o limite de 3 GiB. Após liberar memória, o empacotamento e os testes foram concluídos sequencialmente. Os resultados ficam em `dist/online-installer/validation.json`, `build/guard/online_*.json` e `build/online_installer_e2e`.

O teste positivo NVIDIA e a instalação desta revisão em um computador limpo continuam pendentes. A validação AMD local não comprova suporte de todas as placas ou drivers. Este é um instalador para testes, sem assinatura digital própria ou publicação automática.

Fontes: [instalação oficial AMD/ROCm 7.2.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/windows/install-pytorch.html), [versões oficiais PyTorch](https://pytorch.org/get-started/previous-versions/), [downloads com verificação SHA-256 no Inno Setup](https://jrsoftware.org/inno-setup/documentation/topic_isxfunc_downloadtemporaryfile.htm).
