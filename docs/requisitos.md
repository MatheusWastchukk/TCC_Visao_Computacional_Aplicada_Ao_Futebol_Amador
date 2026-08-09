# Requisitos do Projeto

## Contexto

Este documento transforma o escopo do TCC em requisitos testaveis para o prototipo de deteccao e analise de eventos em videos de futebol. Os requisitos estao divididos em obrigatorios e opcionais para manter o nucleo minimo defensavel do projeto claro durante o desenvolvimento.

## Requisitos Funcionais Obrigatorios

| ID | Requisito | Comprovacao objetiva |
|---|---|---|
| RF-001 | O sistema deve permitir importar um arquivo de video local. | Informar o caminho de um video valido e confirmar que ele e aberto pelo pipeline. |
| RF-002 | O sistema deve validar se o arquivo informado existe e pode ser lido como video. | Executar com arquivo valido, inexistente e invalido, registrando sucesso ou erro compreensivel. |
| RF-003 | O sistema deve extrair metadados basicos do video, incluindo resolucao, FPS, quantidade de frames e duracao aproximada. | Comparar os metadados retornados com `ffprobe` ou OpenCV. |
| RF-004 | O sistema deve processar o video frame a frame. | Registrar a contagem de frames processados em um trecho de teste. |
| RF-005 | O sistema deve detectar jogadores em pelo menos parte dos frames usando modelo de visao computacional. | Gerar resultados estruturados e frames anotados para uma amostra de videos. |
| RF-006 | O sistema deve tentar detectar a bola nos frames processados. | Registrar taxa de frames com bola detectada e exemplos visuais. |
| RF-007 | O sistema deve rastrear a trajetoria da bola ao longo do tempo. | Exportar uma sequencia temporal com frame, timestamp e posicao estimada da bola. |
| RF-008 | O sistema deve permitir definir ou carregar uma regiao do gol. | Carregar um arquivo de configuracao com coordenadas da regiao e validar seu formato. |
| RF-018 | O sistema deve tentar localizar automaticamente a estrutura principal do gol em partidas completas. | Executar o piloto em 3 partidas e registrar precisao, revocacao, F1, IoU e falsos candidatos por hora. |
| RF-019 | O sistema deve gerar janelas candidatas quando a estrutura do gol for confirmada temporalmente. | Registrar intervalos com inicio, fim, confianca media e criterio de agrupamento. |
| RF-020 | O sistema deve retornar `NOT_EVALUABLE` quando nao houver condicao visual suficiente para decidir. | Testar casos com gol invisivel, bola ocluida, baixa confianca ou rastreamento perdido. |
| RF-009 | O sistema deve executar a Abordagem A, baseada em regras espaciais e temporais. | Rodar a abordagem em clipes de teste e exportar resultado, confianca e frame de evidencia. |
| RF-010 | O sistema deve executar a Abordagem B, baseada em classificacao temporal ou alternativa hibrida documentada. | Rodar a segunda abordagem no mesmo formato de entrada e saida da Abordagem A. |
| RF-011 | O sistema deve gerar pelo menos um frame representativo anotado para cada evento candidato. | Conferir arquivo de imagem salvo em `outputs/frames`. |
| RF-012 | O sistema deve exportar metricas de deteccao, rastreamento, evento e desempenho. | Conferir arquivo tabular ou JSON em `outputs/reports`. |
| RF-013 | O sistema deve exibir o resultado em uma interface demonstravel para a banca. | Executar a interface local e completar um fluxo de analise. |

## Requisitos Funcionais Opcionais

| ID | Requisito | Comprovacao objetiva |
|---|---|---|
| RF-014 | O sistema pode permitir anotacao manual assistida da regiao do gol pela interface. | Marcar uma regiao em tela e salvar o arquivo de configuracao. |
| RF-015 | O sistema pode comparar lado a lado as saidas da Abordagem A e da Abordagem B. | Gerar uma tela ou relatorio com resultados das duas abordagens para o mesmo clipe. |
| RF-016 | O sistema pode exportar video curto anotado alem de frames estaticos. | Conferir video anotado em `outputs/videos`. |
| RF-017 | O sistema pode usar aceleracao por GPU quando disponivel. | Registrar backend, hardware e tempo de inferencia em experimento controlado. |

## Requisitos Nao Funcionais Obrigatorios

| ID | Requisito | Comprovacao objetiva |
|---|---|---|
| RNF-001 | O projeto deve funcionar no Windows 11. | Executar instalacao e testes em ambiente Windows 11. |
| RNF-002 | O pipeline minimo deve funcionar em CPU. | Executar um clipe curto sem depender de GPU. |
| RNF-003 | O sistema deve registrar erros de forma compreensivel, sem falhar silenciosamente. | Executar casos invalidos e conferir mensagem/registro de erro. |
| RNF-004 | Os experimentos devem ser reproduziveis. | Versionar configuracoes, dependencias e comandos usados em cada experimento. |
| RNF-005 | As configuracoes do pipeline devem ficar em arquivos externos ao codigo. | Conferir arquivos em `configs/` usados pela execucao. |
| RNF-006 | O sistema nao deve depender de transmissao ao vivo. | Executar somente com arquivos de video locais. |
| RNF-007 | O sistema deve suportar o estudo com partidas completas e permitir amostragem ou recorte para reduzir custo. | Validar metadados de partidas completas e registrar taxa de amostragem usada no piloto. |
| RNF-008 | Videos brutos, clipes grandes, pesos de modelo e saidas geradas nao devem ser versionados por engano. | Conferir `.gitignore` e `git status` antes dos commits. |

## Requisitos Nao Funcionais Opcionais

| ID | Requisito | Comprovacao objetiva |
|---|---|---|
| RNF-009 | O projeto pode oferecer modo de demonstracao offline com resultados pre-processados. | Executar a demonstracao sem internet e sem processamento demorado. |
| RNF-010 | O projeto pode exportar modelos para ONNX se isso facilitar portabilidade. | Gerar arquivo ONNX e executar inferencia com `onnxruntime`. |
| RNF-011 | O projeto pode registrar comparacao entre CPU, GPU AMD e GPU NVIDIA. | Salvar relatorio de tempo e hardware em `experiments/hardware-spike`. |

## Checklist de Revisao

- Cada requisito possui identificador unico.
- Requisitos obrigatorios e opcionais estao separados.
- Cada requisito possui uma forma objetiva de comprovacao.
- O escopo minimo permanece alinhado ao plano do TCC.
