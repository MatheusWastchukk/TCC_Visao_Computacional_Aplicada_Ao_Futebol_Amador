# Métricas nos detalhes do processamento

A faixa mantém os seis indicadores, mas aparece apenas ao expandir **Detalhes do processamento**. O painel começa fechado em cada sessão; atualizações de progresso preservam a escolha de abrir ou fechar. A lista duplicada de métricas no cabeçalho foi removida.

## Significado dos valores

- **CPU da análise:** uso do processo Python que executa a análise, normalizado pela capacidade total dos processadores lógicos. Em oito processadores lógicos, ocupar integralmente um deles corresponde a aproximadamente 12,5%. O nome anterior, “CPU do backend”, não descrevia isso claramente para o usuário. A leitura não inclui a interface Electron nem processos separados do FFmpeg. GPU também exige CPU para leitura e preparação de quadros.
- **Memória RAM:** memória residente desse mesmo processo, em MB e como percentual da RAM física total.
- **Uso de GPU:** atividade atual da placa selecionada. No Windows, usa contadores WDDM de todos os processos dessa placa; portanto, também inclui outros aplicativos.
- **Memória da GPU:** uso total da placa informado por `torch.cuda.mem_get_info`, separado da taxa de atividade. Percentual de VRAM ocupada não é percentual de processamento.
- **Tempo:** duração da análise, conforme o estado da sessão.

## Coleta AMD no Windows

A leitura antiga dependia exclusivamente de `torch.cuda.utilization`. Falhas nessa API deixavam a leitura indisponível. O novo monitor usa as APIs nativas DXGI e PDH, sem instalar NVML, ROCm SMI ou outros pacotes.

DXGI associa o nome da GPU selecionada ao seu LUID. A associação recusa placas idênticas ambíguas, em vez de usar o índice errado. Os contadores `GPU Engine(*)\Utilization Percentage` são somados entre processos por motor físico da placa; o indicador usa o motor mais ocupado, limitado a 100%, conforme o critério descrito pela Microsoft para o Gerenciador de Tarefas. Não soma placas ou motores diferentes.

A consulta PDH é persistente e amostrada em uma thread de segundo plano, aproximadamente uma vez por segundo. A API consulta o último valor sem esperar uma amostragem. A primeira leitura aparece como “Aguardando leitura…”. Zero é uma leitura válida; ausência de dados não é convertida em zero. Erros temporários são tentados novamente. A consulta limita o buffer de contadores a 16 MiB.

O PyTorch permanece como alternativa quando a leitura Windows está indisponível ou em outras plataformas. O JSON das métricas identifica fonte, escopo e estado da leitura. O modo CPU não inicia a coleta WDDM.

## Validação em 19/11/2026

- Cinco testes novos verificaram agregação, exclusão de outra GPU, zero real, associação ambígua, persistência da consulta e recuperação de erro. Com a regressão existente da CPU, passaram seis testes Python.
- A leitura nativa na Radeon RX 9060 XT local identificou o adaptador correto e retornou amostras válidas. A coleta em segundo plano também passou com RedirectionGuard ativo, sem iniciar modelos ou análise de vídeo. A amostra de 0,2% observada nessa checagem não é um benchmark de inferência.
- O transformador JSX do Rolldown validou os arquivos React. A renderização estática confirmou uma faixa única dentro de `<details>`, sem atributo `open` inicial, com o rótulo novo e preservação de 0%.
- A validação visual no navegador, a integração com o backend empacotado e a atualização do instalador/aplicativo ainda dependem de memória disponível e do encerramento da aplicação em uso. O executável distribuído anteriormente ainda não contém estes ajustes.

Nenhum vídeo TRAIN/TEST foi processado para estes ajustes. Os resultados científicos congelados não foram modificados.

## Empacotamento em 20/11/2026

As alterações foram incluídas no novo instalador 0.4.3 de 404.628.895 bytes. A cópia empacotada abriu em CPU com perfil isolado e sessão fictícia, sem processar vídeo. A validação por navegador confirmou os detalhes fechados inicialmente, conteúdo oculto, rótulo “CPU da análise” e conteúdo visível após o clique; a captura de tela foi inspecionada. Os módulos de coleta/backend foram comparados com os fontes dentro do executável e os arquivos da interface com o ASAR. O teste real das métricas durante processamento AMD/NVIDIA nesta reconstrução permanece pendente. Evidências em `build/launcher_043_cpu_cpu`, `build/guard/release_20261120_*.json` e `dist/online-installer/validation.json`.

Referências: [GPU no Gerenciador de Tarefas](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/), [PDH em inglês independente do idioma do Windows](https://learn.microsoft.com/windows/win32/api/pdh/nf-pdh-pdhaddenglishcounterw), [leitura de contadores com múltiplas instâncias](https://learn.microsoft.com/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw), [identificação de adaptadores DXGI](https://learn.microsoft.com/windows/win32/api/dxgi/ns-dxgi-dxgi_adapter_desc).
