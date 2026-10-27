# Requisitos e critérios de aceite do TCC

## Escopo do trabalho

O projeto atende ao título oficial **Detecção e Análise de Eventos em Vídeos de Futebol Utilizando Técnicas de Visão Computacional**. O escopo efetivamente avaliado é a mineração e priorização de trechos potencialmente relacionados a gols para revisão humana. O sistema não confirma gols automaticamente.

Pergunta de pesquisa: em partidas completas do corpus, quanto da cobertura dos timestamps de gols catalogados pode ser preservada por políticas fixas, selecionadas em TRAIN e aplicadas em TEST, sob diferentes tetos de minutos únicos por partida?

## Requisitos da contribuição experimental

| ID | Requisito | Critério de evidência | Estado |
|---|---|---|---|
| EXP-01 | Manter separados Candidate Pool, Nested LOMO, seleção de políticas e avaliação final. | População, protocolo, unidade e interpretação informados para cada resultado. | Atendido no artigo; tabela-síntese adicionada. |
| EXP-02 | Avaliar o procedimento de seleção por partida e orçamento em TRAIN. | Nested LOMO 25/25, 5.742 grupos e 152 eventos avaliáveis, com resultados por orçamento. | Executado e documentado. |
| EXP-03 | Avaliar políticas experimentais fixas em TEST sem retuning. | Freeze e manifests rastreáveis; timestamps humanos congelados antes da inferência; cobertura e minutos únicos reportados. | Executado uma vez; TEST fechado para seleção. |
| EXP-04 | Definir cobertura de evento de modo reproduzível. | Timestamp humano pertence a pelo menos um intervalo selecionado; inclusão fechada e tolerância informadas. | Definido; tolerância adicional zero em TEST. |
| EXP-05 | Distinguir cobertura de precisão e de esforço humano. | Limitações explícitas no resumo, método, discussão e conclusão. | Atendido; não há precisão ou tempo humano medidos. |
| EXP-06 | Documentar limites da bola e do rastreamento. | Diferenciar proposta, identidade, continuidade e interpretação de evento, com evidência visual/quantitativa. | Atendido no artigo e relatórios. |
| EXP-07 | Registrar decisões de promoção e não promoção de métodos. | Relatórios identificam resultados exploratórios, confirmatórios, diagnósticos e não promovidos. | Documentado no histórico experimental. |

## Requisitos da aplicação de revisão

| ID | Requisito | Critério de evidência | Estado |
|---|---|---|---|
| APP-01 | Abrir vídeos locais e reportar entradas inválidas com erro visível. | Cenários E2E de vídeo válido e MP4 inválido. | Exercitado manualmente. |
| APP-02 | Permitir navegação, reprodução e inspeção de candidatos. | Player, seek, atalhos, storyboard e navegação exercitados. | Exercitado manualmente. |
| APP-03 | Registrar decisões humanas e notas em sessão local. | Persistência, retomada e edição exercitadas. | Exercitado manualmente. |
| APP-04 | Exportar dados e trechos da sessão. | Exportações HTML, CSV, JSON e MP4 exercitadas. | Exercitado manualmente. |
| APP-05 | Identificar o pipeline operacional e guardar rastreabilidade da sessão. | Versão, pipeline, modo, dispositivo e artefatos visíveis/conferidos. | Exercitado; uso percentual da GPU indisponível no ambiente. |
| APP-06 | Oferecer undo persistente da última classificação e cancelamento de análise em execução com limpeza da saída parcial. | E2E-06/E2E-07, build e verificações funcionais. | Implementado; ambos os fluxos foram retestados manualmente e confirmados pelo usuário em 25/10/2026. |

A matriz registra 13 cenários aprovados conforme as evidências disponíveis. O usuário retestou E2E-06 e E2E-07 manualmente em 25/10/2026 e confirmou ambos. Isso não constitui estudo formal de usabilidade nem medição de latência do cancelamento.

## Requisitos de reprodução e distribuição

| ID | Requisito | Critério de evidência | Estado |
|---|---|---|---|
| REP-01 | Identificar commit, versão de ambiente, dependências, configurações e manifests. | Manifesto/checklist de entrega reproduzível com hashes e localização dos artefatos. | Manifesto preliminar criado; snapshot e hashes finais pendentes. |
| REP-02 | Distinguir reprodução de métricas da repetição de inferência. | Instruções e recursos necessários separados para cada nível. | Distinção documentada em `docs/entrega_reprodutivel.md`. |
| REP-03 | Declarar artefatos não incluídos. | Vídeos, pesos, ambientes volumosos/licenciados e requisitos de runtime listados. | Exclusões e build legado documentados; licenças e pacote final pendentes. |
| REP-04 | Não reabrir TEST para desenvolver ou escolher políticas. | Congelamento, hashes e separação entre código de avaliação e ajuste. | Regra vigente. |

## Fora do que foi demonstrado

- Precisão/recall geral de detecção automática de gols ou taxa de falsos candidatos confirmada por especialistas.
- Economia de tempo humano ou melhoria de decisão em estudo controlado.
- Generalização para domínios de câmera/competição não representados no corpus.
- Empacotamento portátil autocontido ou execução confiável em CPU sem GPU.
- Integração das políticas V3 experimentais nos modos V4 da interface.
