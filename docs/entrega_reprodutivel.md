# Plano de reprodução do repositório e das evidências

**Estado em 26/10/2026:** rota inicial escolhida para reproduzir as métricas congeladas; pacote final ainda não materializado e distribuição bloqueada por revisão de direitos/licenças pendente.
**Commit base identificado:** `65745a5553da7b7f1563b410e3651e268bccb228` (HEAD atual; working tree dirty).
**Árvore de trabalho:** contém alterações locais; o manifesto descreve este estado como rascunho e não como release.

## Objetivo

O artigo do TCC é desenvolvido no Overleaf. Este plano cobre somente o repositório local, seu ambiente de execução e os artefatos de código/dados usados para reproduzir resultados; arquivos LaTeX, PDF, template e figuras do artigo não são requisitos deste pacote.

Entregar à banca uma cópia que permita compreender o código e reproduzir a interpretação dos resultados sem distribuir vídeos, pesos ou ambientes grandes sem autorização. Separar duas formas de reprodução:

1. **Rota escolhida para a entrega inicial: reprodução das métricas publicadas a partir dos artefatos congelados.** Usa CSVs e resumos da avaliação já executada; não roda inferência nem altera políticas. Uma cópia temporária mínima reproduziu os resultados em 26/10/2026: `goal_coverage_test_v1.csv` e `paired_mode_gains_test_v1.csv` foram idênticos; `per_match_test_v1.csv` e os campos numéricos do resumo coincidiram com tolerância absoluta de `1e-12`. Seis campos de ponto flutuante em duas partidas diferiram apenas por arredondamento de máquina, abaixo de `4e-15`. Nenhum vídeo, peso ou embedding foi usado. O script passou a aceitar BOM UTF-8 nos manifests JSON.
2. **Reexecução completa.** Exige os vídeos locais, pesos exatos, hardware/backend compatível, dependências, FFmpeg, configurações e scripts da avaliação. É opcional e depende de acesso autorizado aos dados e modelos.

A reexecução de TEST, se necessária apenas para reprodução, não poderá ser usada para retuning, comparação de políticas novas ou escolha de vencedor. A verificação em cópia mínima reproduz as métricas derivadas dos artefatos salvos; não valida a inferência completa nem uma instalação limpa do aplicativo.

## Pacote legado encontrado

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
- Instalador portátil atual da aplicação Electron: não há configuração electron-builder confirmada. O build Vite não empacota Python, pesos, ambiente ROCm nem toda a cadeia de FFmpeg.

Para cada item excluído, registrar nome ou categoria, motivo e instrução de obtenção/acesso quando disponível. Não prometer que a aplicação funciona em máquina limpa ou em CPU sem GPU sem executar essa verificação.

## Matriz de reprodução

| Etapa | Artefatos mínimos | Ação permitida | Critério de aceite |
|---|---|---|---|
| Conferir métricas congeladas | CSVs/JSONs de TEST, scripts de agregação e hashes | Recalcular os resumos e comparar com os valores de referência registrados | Contagens e arredondamentos conferidos; TEST não usado para ajuste |
| Reproduzir Nested TRAIN | Dados de features autorizados, script, modelos/configuração registrados e relatório | Reexecutar protocolo TRAIN conforme comandos documentados | População, folds, parâmetros e resultados conferidos |
| Reexecutar TEST | Vídeos TEST autorizados, pesos exatos, catálogo já congelado, configuração e ambiente registrados | Uma reprodução dos modos congelados, sem retuning | Hashes/configuração correspondem ao freeze e métricas são determinísticas |
| Abrir a aplicação | Python, dependências, pesos, FFmpeg, Node/npm e instrução versionada | Iniciar fluxo V4 com vídeo de demonstração permitido | Sessão local processa, permite revisão e exporta conforme E2E documentado |

## Manifesto preliminar

O manifesto registra hashes dos códigos de dependências e evidências congeladas. Ele descreve um inventário preliminar do repositório, não um pacote final nem arquivos do Overleaf.

## Pendências antes de gerar o pacote
- [x] Reconciliar `requirements.txt` e `pyproject.toml` com o ambiente Python/ROCm observado; registrar a instalação ROCm. Ultralytics foi movido para o grupo experimental opcional por não ser importado pelo aplicativo V4.
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

Os achados e limites estão em [`auditoria_licencas_pacote.md`](auditoria_licencas_pacote.md). Não há licença de distribuição do próprio projeto no checkout. Vídeos, pesos, dados externos e artefatos derivados de TEST seguem excluídos até comprovar autorização. A remoção de Ultralytics da receita base reduz o escopo dessa dependência para scripts experimentais, mas não autoriza distribuir esses scripts, pesos ou derivados.
## Risco conhecido

O estado atual não é um pacote portátil validado da aplicação Electron. O backend operacional foi construído em ambiente local Windows/AMD e há dependência de pesos e executáveis externos. Há uma saída antiga PyInstaller baseada em PySide6, mas ela não corresponde ao fluxo E2E atual. Se não houver tempo para teste limpo, a entrega deve ser descrita como pacote de código e artefatos para reprodução em ambiente compatível, sem alegar executável independente.
