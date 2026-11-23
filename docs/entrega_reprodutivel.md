# Plano de reprodução do repositório e das evidências

**Revisão documental:** 23/11/2026. A rota inicial continua sendo o recálculo das métricas congeladas; a identificação da versão e as instruções diretas estão no [guia de reprodução para a banca](guia_reproducao_banca.md).
**Base de código conferida nesta revisão:** `265b95e5e35a6ab4816a00de7d0b2023c9a6303b`, com árvore limpa antes dos ajustes documentais de 23/11. Esses ajustes ainda precisam integrar o snapshot efetivamente entregue.
**Estado do pacote:** o manifesto de 26/10 é um inventário histórico `DRAFT_NOT_A_RELEASE`, não um manifesto da árvore atual nem uma autorização de distribuição. A revisão de direitos/licenças e o manifesto final da cópia entregue permanecem pendentes.

## Objetivo

O artigo do TCC é desenvolvido e compilado no Overleaf; há uma cópia local do fonte em `docs/TCC_Matheus_Wastchuk.tex`. Este plano cobre o repositório, seu ambiente de execução e os artefatos usados para reproduzir resultados. O recálculo das métricas não depende do LaTeX, PDF, template ou figuras; a conferência editorial depende do PDF final entregue.

Entregar à banca uma cópia que permita compreender o código e reproduzir a interpretação dos resultados sem distribuir vídeos, pesos ou ambientes grandes sem autorização. Separar duas formas de reprodução:

1. **Rota escolhida para a entrega inicial: reprodução das métricas publicadas a partir dos artefatos congelados.** Usa CSVs e resumos da avaliação já executada; não roda inferência nem altera políticas. Uma cópia temporária mínima reproduziu os resultados em 26/10/2026: `goal_coverage_test_v1.csv` e `paired_mode_gains_test_v1.csv` foram idênticos; `per_match_test_v1.csv` e os campos numéricos do resumo coincidiram com tolerância absoluta de `1e-12`. Seis campos de ponto flutuante em duas partidas diferiram apenas por arredondamento de máquina, abaixo de `4e-15`. Nenhum vídeo, peso ou embedding foi usado. O script passou a aceitar BOM UTF-8 nos manifests JSON.
2. **Reexecução completa.** Exige os vídeos locais, pesos exatos, hardware/backend compatível, dependências, FFmpeg, configurações e scripts da avaliação. É opcional e depende de acesso autorizado aos dados e modelos.

A reexecução de TEST, se necessária apenas para reprodução, não poderá ser usada para retuning, comparação de políticas novas ou escolha de vencedor. A verificação em cópia mínima reproduz as métricas derivadas dos artefatos salvos; não valida a inferência completa nem uma instalação limpa do aplicativo.

## Aplicação atual e pacote legado

A aplicação Electron/React + FastAPI declara versão `0.4.3` em `desktop/package.json`. Há receitas de empacotamento e instaladores documentados em [build CPU](build_cpu.md) e [instalador online 0.4.3](instalador_online_043.md). O segundo documento identifica por SHA-256 o instalador reconstruído em 20/11 e distingue as verificações dessa revisão dos testes históricos. A existência do build não demonstra compatibilidade universal nem valida os modos V4 pelo resultado TEST V3.

A mineração V4 carregada dinamicamente importa Ultralytics, incluído em `requirements-cpu.txt`. A classificação de outubro como dependência exclusivamente experimental foi corrigida em novembro; consulte a [auditoria de licenças](auditoria_licencas_pacote.md).

Existe uma saída PyInstaller antiga em `release/GoalEventReview/`, construída em 17/09/2026 pelo `release/GoalEventReview.spec`. O spec usa PySide6 e o ponto de entrada `app/desktop_review_app.py`, além de incluir caminhos absolutos deste computador e pesos/modelos da arquitetura anterior. A interface validada em outubro é Electron/React + FastAPI. Portanto, a pasta de release não é evidência de que a aplicação atual esteja empacotada ou validada como executável portátil. **Não distribuir o build legado como entrega atual** sem auditar versão, conteúdo e funcionamento; o manifesto de entrega o registra como artefato legado excluído.

## Conteúdo a incluir

### Código e descrição do sistema

- Código versionado do pipeline, aplicação, scripts experimentais relevantes e testes existentes.
- Requisitos, rastreabilidade e instruções de entrega atualizados.
- Arquivo de dependências Python `requirements.txt`, dependências alternativas somente quando usadas, `desktop/package.json` e `desktop/package-lock.json`.
- Instruções de setup, comandos e versões de Python, Node/npm, PyTorch, TorchVision, ROCm/DirectML, OpenCV e FFmpeg realmente usados na reprodução.
- Identificação clara dos modos experimentais V3 e dos modos operacionais V4.

### Evidências e manifests

- Freeze V3 em `experiments/ball/REPORT87-PD-SPRINT-FINAL-FREEZE-20261011/final_freeze_manifest_v3.json`.
- Configuração e manifests de seleção/inferência V3 no diretório REPORT98 referido pelo freeze.
- Protocolo e manifesto de freeze do catálogo cego TEST.
- Relatório final em `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/test_v3_results_report.md`.
- Arquivos finais de métricas: `summary_test_v1.json`, `per_match_test_v1.csv`, `paired_mode_gains_test_v1.csv` e `goal_coverage_test_v1.csv`.
- Resultados Nested, seleção das políticas, protocolo de avaliação e scripts de agregação usados para produzir as tabelas.
- Hash SHA-256 para cada arquivo entregue e identificação da versão/commit correspondente.

## Itens não incluídos por padrão

- Vídeos brutos das 42 partidas, clipes derivados e dados de terceiros.
- Pesos de modelos e checkpoints, até confirmar licença, tamanho, hashes e autorização de distribuição.
- Ambientes virtuais, caches ROCm/MIOpen, outputs regeneráveis, sessões locais do Electron, logs pessoais e artefatos temporários.
- Embeddings ou outras matrizes grandes; incluir somente se forem necessários à reprodução escolhida e se houver autorização.
- O build PyInstaller legado em `release/GoalEventReview/`, que não corresponde à aplicação Electron V4 validada em E2E.
- Instaladores e runtimes da aplicação Electron não integram automaticamente a cópia de código/métricas. As receitas atuais usam electron-builder e empacotamento do backend; a seleção do instalador deve registrar versão, SHA-256, componentes incluídos e os limites de validação documentados. O build Vite isolado compila somente a interface.

Para cada item excluído, registrar nome ou categoria, motivo e instrução de obtenção/acesso quando disponível. Não prometer que a aplicação funciona em máquina limpa ou em CPU sem GPU sem executar essa verificação.

## Matriz de reprodução

| Etapa | Artefatos mínimos | Ação permitida | Critério de aceite |
|---|---|---|---|
| Conferir métricas congeladas | CSVs/JSONs de TEST, scripts de agregação e hashes | Recalcular os resumos e comparar com os valores de referência registrados | Contagens e arredondamentos conferidos; TEST não usado para ajuste |
| Reproduzir Nested TRAIN | Dados de features autorizados, script, modelos/configuração registrados e relatório | Reexecutar protocolo TRAIN conforme comandos documentados | População, folds, parâmetros e resultados conferidos |
| Reexecutar TEST | Vídeos TEST autorizados, pesos exatos, catálogo já congelado, configuração e ambiente registrados | Uma reprodução dos modos congelados, sem retuning | Hashes/configuração correspondem ao freeze e métricas são determinísticas |
| Abrir a aplicação | Python, dependências, pesos, FFmpeg, Node/npm e instrução versionada | Iniciar fluxo V4 com vídeo de demonstração permitido | Sessão local processa, permite revisão e exporta conforme E2E documentado |

## Manifesto preliminar

`entrega_reprodutivel_manifesto_v1.json` e seu checksum registram o inventário preliminar de 26/10, baseado em `65745a5`, com snapshot identificado como `2fddf5a`. Foram preservados como evidência histórica; seus hashes e descrições de dependências não representam a árvore de novembro. Para a rota atual de métricas, use o [guia da banca](guia_reproducao_banca.md). O manifesto final da cópia entregue ainda deve ser gerado sobre os arquivos efetivamente selecionados.

## Pendências antes de gerar o pacote
- [x] Corrigir a classificação de Ultralytics: é dependência efetiva da mineração V4 carregada dinamicamente e consta de `requirements-cpu.txt`; as receitas e os limites de ambiente estão em `ambiente_reproducao.md`, `build_cpu.md` e `instalador_online_043.md`.
- [ ] Consolidar o estado atual do Git em snapshot auditado, preservando mudanças locais pertinentes.
- [ ] Confirmar licença do código do projeto e autorizações para os CSVs derivados, pesos e cada vídeo/dado externo.
- [x] Escolher métricas congeladas como rota inicial e reproduzir o cálculo em cópia temporária mínima, sem retuning de TEST.
- [x] Validar a rota de métricas dentro de um snapshot temporário da árvore-fonte; `npm ci`, `npm ls`, audit sem vulnerabilidades, build Vite e agregação dos CSVs passaram.
- [ ] Reproduzir a receita Python/ROCm e grupos opcionais numa cópia limpa; o snapshot validado usou o Python do ambiente local para a etapa de métricas.
- [ ] Reproduzir a rota escolhida dentro de um snapshot final auditado; a cópia temporária atual não é o pacote final.
- [ ] Gerar manifesto final com caminhos, tamanhos, SHA-256 e estado de inclusão/exclusão.
- [ ] Confirmar que scripts e caminhos documentados existem no pacote final.
- [ ] Registrar a exclusão do embedding derivado de TEST e verificar se algum procedimento de reprodução depende dele.

## Auditoria preliminar de licenças e direitos

Os achados e limites estão em [`auditoria_licencas_pacote.md`](auditoria_licencas_pacote.md). A auditoria registra pendências de licença do código autoral e de autorização para vídeos, pesos e derivados TEST. Ultralytics integra a mineração V4 e deve ser considerado na revisão da distribuição da aplicação. Builds e testes locais não encerram essa revisão. A presença de uma entrada no guia de reprodução não autoriza sua redistribuição.
## Risco conhecido

A reprodução das métricas salvas, a reexecução da inferência e a instalação da aplicação são verificações distintas. Há empacotamento CPU e instalador 0.4.3 documentados, com testes e limites específicos por revisão; não se deve extrapolá-los para outro hardware ou para todos os componentes GPU. A saída antiga PySide6 permanece excluída. Descreva a entrega conforme o conteúdo efetivamente fornecido e mantenha explícitas as pendências de ambiente e distribuição.
