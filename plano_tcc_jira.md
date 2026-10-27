## Remoção da caixa de informações dos clipes (08/10/2026)

- [x] Clipes novos e exportações sem overlay; marcações humanas persistem nas anotações sem re-renderizar o MP4.
- [x] Sessões antigas migram sob demanda na reprodução/download/compilação, do vídeo original e com o mesmo nome de clipe.
- [ ] Confirmar manualmente no Electron clipes novos e migração de sessão antiga.


## Corrigida a inicialização do renderer Electron (08/10/2026)

- [x] O Electron verifica se o Vite está acessível; quando não está, abre o build local em desktop/dist/index.html.
- [x] O comando npm start recompila a interface antes de abrir o Electron.
- [x] Node syntax check e build passaram; a abertura e uso manual foram cobertos depois pela matriz E2E 13/13 (out/2026).


## Corrigida tela branca da interface React (08/10/2026)

- [x] Identificadas referências a startAnalysis e openSession sem definição durante o primeiro render; adicionadas junto aos endpoints existentes da API.
- [x] Restaurado o carregamento/polling de estado e clipes para análises novas e retomadas; definida a função refreshClips usada no tratamento de erro.
- [x] Build Vite e verificação de sintaxe do processo Electron passaram; falta confirmar a abertura manual.


## Correção da inicialização no comando de desenvolvimento (08/10/2026)

- [x] O comando dev agora remove ELECTRON_RUN_AS_NODE do processo Electron e passa a URL real do Vite; funciona mesmo se a porta padrão estiver ocupada.
- [x] Reprodução do subprocesso confirmou API local respondendo HTTP 200 na versão 0.4.0. Build Vite e verificações de sintaxe passaram.
- [x] Registro historico: uma execucao grafica falhou; a inicializacao posterior foi aprovada em E2E-01. A telemetria de uso GPU segue indisponivel.

## Consolidação dos relatórios 49–92 — 13/10/2026

- [x] Inventariados os 44 documentos numerados 49–92 em `docs/plans/`.
- [x] Consolidada a evolução metodológica, resultados, decisões, limitações e artefatos no relatório [93 — consolidação integral dos relatórios 49 a 92](docs/plans/93_relatorio_consolidado_integral_relatorios_49_a_92.md).
- [x] Separados resultados TRAIN/LOMO, confirmações VALIDATION, amostras diagnósticas, Match 43 externo e implementação de software.
- [x] Preservada a distinção entre pool e shortlist, minutos únicos de vídeo e tempo humano, rótulos humanos e weak labels.
- [x] Registro historico: uma execucao grafica falhou; a inicializacao posterior foi aprovada em E2E-01. A telemetria de uso GPU segue indisponivel.
- [x] E2E manual ponta a ponta concluido: matriz com 13/13 aprovados; estudo de tempo humano permanece nao realizado.


## Nested LOMO full-pool em 25 partidas TRAIN — 14/10/2026

- [x] Recuperados 370 grupos `net_motion` ausentes e montada população comum de 5.742 grupos, 25 partidas e 152 gols avaliáveis; os 5.372 registros históricos foram preservados e conferidos.
- [x] Executado Nested LOMO 25/25 no grid e orçamento congelados; cobertura, minutos, bootstrap pareado (10.000 reamostragens, seed 901006) e Pareto registrados no Relatório 95.
- [x] Selecionados modos fixos candidatos V2; configurações e métricas in-sample estão separados das estimativas Nested OOF.
- [x] Criado snapshot intermediário v46 com 172 hashes; o preflight histórico passou e validou as evidências anteriores do sprint.
- [x] Histórico superado: freeze V3 final e avaliação TEST V3 concluídos; não reexecutar preflight para alterar o protocolo.
- [x] TEST V3 ja foi avaliado uma vez sob protocolo congelado; manter fechado para qualquer novo ajuste.
- Relatórios: [Plano 94](docs/plans/94_plano_execucao_nested_lomo_fullpool_25_train_14-10-2026.md) e [Resultados 95](docs/plans/95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md).

## Avaliação final V3 no TEST — 19/10/2026

- [x] Concluída a anotação cega dos 57 eventos em nove partidas; catálogo congelado com SHA-256 registrado. O `match_40`, gol 006, foi revisitado porque o usuário informou que o gol passou despercebido na primeira passagem; timestamp confirmado e preservado.
- [x] Registrado o protocolo antes da inferência e executados os três modos V3 congelados uma única vez, sem consulta ao catálogo durante o ranking.
- [x] Cobertura TEST: FAST_V3 45/57 (78,95%); BALANCED_V3 50/57 (87,72%); HIGH_RECALL_V3 54/57 (94,74%). Cobertura não é precisão automática de gols.
- [x] Verificado o custo contra os tetos congelados: zero das nove partidas acima de 30/40/58 min por FAST/BALANCED/HIGH_RECALL.
- [x] Publicados resultados por gol, partida, diferenças pareadas, minutos únicos, execução, hashes e ressalvas em [relatório completo da avaliação V3 TEST](experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/test_v3_results_report.md).
- [x] Corrigido um erro de parsing de `False` em texto no pós-processamento; métricas recalculadas sem repetir inferência. Resultados finais em `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/metrics_v1/`.
- [x] Atualizar artigo e discussão com os resultados TEST separados das estimativas Nested OOF TRAIN; nenhuma política foi escolhida, retreinada ou ajustada com base no TEST. Manuscrito atualizado em 23/10/2026.
- [x] Revisão editorial de validade fica fora do escopo do repositório local; artigo no Overleaf.
- [x] Validar manualmente o fluxo completo da aplicação Electron e registrar resultados E2E; concluído em 22/10/2026 com exceções de escopo e limitações documentadas abaixo.

## Preparação da atualização do artigo e aplicação — 20/10/2026

- [x] Criado o [Relatório 102 — atualização do artigo após avaliação V3 em TEST](docs/plans/102_relatorio_atualizacao_artigo_resultados_test_v3_20-10-2026.md), com mapa de seções, tabela, textos acadêmicos sugeridos, limitações e fontes rastreáveis.
- [x] Confirmado o build da interface: `npm.cmd run build --prefix desktop` passou (Vite 8.3.0; 1.877 módulos). O build não substitui teste de uso da aplicação.
- [x] Criado o [modelo da matriz E2E](docs/plans/application_e2e_matrix_2026-10-20.csv), com 13 cenários previstos para uso de vídeos TRAIN.
- [x] Fonte atual localizado em `docs/TCC_Matheus_Wastchuk.tex`; resumo, Abstract, metodologia, resultados, discussão, limitações e conclusão atualizados em 23/10/2026 conforme o Relatório 102.
- [x] Executar a matriz E2E na interface Electron; os 13 cenários foram avaliados e registrados em `docs/plans/application_e2e_matrix_2026-10-20.csv` até 22/10/2026.
- [x] Fora do escopo local: compilacao e revisao do artigo ocorrem no Overleaf.

### E2E-01 — inicialização Electron/API (20/10/2026)

- [x] Primeira tentativa interativa registrou alerta de falha da API; após o alerta, `/health` respondeu `status=ok`, `version=0.4.0`. Evidência detalhada na matriz E2E.
- [x] Corrigida a espera do Electron: timeout ampliado de 10 para 60 segundos; a mensagem agora informa se o Python falhou ao iniciar ou encerrou. `node --check desktop/electron/main.cjs` e build Vite passaram.
- [x] Reteste após reinicialização confirmado pelo usuário: o alerta parou de aparecer. E2E-01 aprovado; a matriz completa foi concluída em 22/10/2026.

### Fechamento da matriz E2E (22/10/2026)

- [x] Aprovados: inicialização/análise, player e atalhos, revisão e filtros, persistência, sessão antiga, modal de erro, exportações, modos/pipeline e rastreabilidade dos artefatos.
- [x] Registro histórico de 22/10: E2E-06 aprovado com observação; desfazer ainda não estava implementado. Atualização de 25/10: recurso implementado e reteste manual aprovado pelo usuário (ver seção abaixo e matriz E2E).
- [x] Registro histórico de 22/10: E2E-07 avaliado antes da implementação do cancelamento. Atualização de 25/10: recurso implementado e reteste manual aprovado pelo usuário (ver seção abaixo e matriz E2E).
- [x] Registrada a telemetria de percentual de uso da GPU indisponível no ambiente AMD; dispositivo e memória da GPU foram identificados.
- [x] Fora do escopo do repositorio local: revisao editorial do manuscrito ocorre no Overleaf.

## Fechamento editorial e pacote de entrega reproduzível — 23/10/2026

- [x] Manter o título institucional vigente e delimitar no texto a pergunta de pesquisa, contribuição e definição de cobertura.
- [x] Inserir literatura acadêmica sobre action spotting temporal e seleção de dados para anotação; distinguir essas tarefas do objetivo deste estudo.
- [x] Atualizar a duração arredondada do corpus no resumo para 61,7 horas e manter a duração exata de 61h41min35s na tabela.
- [x] Corrigir a auditoria do Relatório 103 com a duração de TEST de 11h47min48s, a razão agregada de vídeo e os ganhos pareados confirmados nos CSVs.
- [x] Atualizar README, requisitos e matriz de rastreabilidade para o escopo final.
- [x] Criar o plano do pacote reproduzível com separação entre reprodução de métricas e reexecução integral, inventário e exclusões.
- [x] Criar o plano condicional do piloto de revisão humana TRAIN-only.
- [x] Revisar o Relatório 103 nas seções executivas/checklist para remover as pendências numéricas já resolvidas.
- [x] Registrado como fora do escopo local: artigo, template article-config e PDF são mantidos no Overleaf.
- [x] Fora do escopo local: citacoes, referencias e layout sao revisados no Overleaf.
- [x] Corrigir no Relatório 102 os ganhos pareados e a descrição da avaliação TEST, alinhando-os aos CSVs.
- [ ] Revisar arquivos alterados, selecionar staging e registrar SHA; ver checklist de reprodutibilidade ao final.
- [ ] Avaliar até 31/10 se o piloto humano tem aprovação institucional, participantes e tempo sem afetar o artigo; se não, registrá-lo como trabalho futuro.

### Entrega reproduzível — inventário e limites conhecidos

- [x] Conferir ambiente de dependências Python e Electron, manifests V3/TEST e hashes das principais métricas.
- [x] Identificar a saída PyInstaller de 17/09 como build legado PySide6, separado do Electron V4 validado em E2E.
- [x] Registrar artefatos que não acompanham a entrega por padrão: vídeos, pesos, ambientes locais, outputs grandes e build legado.
- [x] Criar manifesto preliminar de entrega; ele registra o commit base, estado dirty, hashes e bloqueios ainda em aberto.
- [x] Fora do escopo local: template, figuras e compilação do artigo permanecem no Overleaf.
- [ ] Revisar autorizações/licenças antes de incluir dados, vídeos e pesos externos no pacote.
- [x] Escolhida a rota inicial por métricas congeladas; cálculo reproduzido num snapshot limpo sem inferência, vídeos ou pesos. Inferência completa fica opcional.



### Correção dos pontos pendentes da aplicação (25/10/2026)

- [x] Implementado desfazer da última decisão: restaura rótulo e estado anteriores; histórico persistido em `session.json` por sessão; endpoint `POST /sessions/{id}/undo` e botão na interface.
- [x] Implementado cancelamento da análise: botão e endpoint, checkpoints cooperativos na mineração/pontuação visual, interrupção do FFmpeg e remoção da pasta `analysis` parcial.
- [x] `python -m compileall` dos módulos alterados passou; `npm run build` passou.
- [x] Checagens funcionais isoladas passaram: sequência de duas classificações e dois undos; worker de análise simulado interrompido em checkpoint e saída temporária removida.
- [x] Retestes manuais de E2E-06 e E2E-07 realizados; usuário confirmou que desfazer e cancelar funcionaram em 25/10/2026.
- [ ] Medir futuramente a latência do cancelamento por etapa/equipamento; esse dado não foi coletado no reteste funcional.

A matriz e o Relatório 103 agora registram os dois retestes manuais como aprovados, sem inferir latência ou condições de hardware que não foram documentadas.


### Ajuste do escopo do repositório local (25/10/2026)

- [x] Registrado que o artigo é desenvolvido no Overleaf; arquivos LaTeX/PDF/templates/figuras não são escopo do pacote local.
- [x] README e plano de reprodução delimitam o pacote a código, ambiente e evidências locais.
- [x] Corrigido o manifesto JSON preliminar e registradas as versoes observadas do ambiente.
- [ ] Instalar em ambiente limpo e confirmar a receita Python/ROCm documentada; ainda não executado.
- [ ] Criar snapshot/commit local após auditoria seletiva de todos os arquivos pendentes.


### Revisão do repositório e ambiente (26/10/2026)

- [x] Working tree inventariado; fonte do artigo/Overleaf e embedding derivado permanecem locais e ignorados.
- [x] Revisados os diffs da aplicação, interface, requisitos e rastreabilidade; corrigida entrada duplicada no manifesto do freeze intermediário.
- [x] Registros antigos de freeze/TEST/E2E atualizados como concluídos ou históricos superados; pendência de clipes antigos mantida aberta por falta de confirmação.
- [x] Python 3.12 selecionado; dependências base alinhadas ao ambiente ROCm observado e ferramentas/experimentos separados em grupos opcionais.
- [x] Dependências diretas do desktop fixadas às versões do lockfile; `package.json` e `package-lock.json` sincronizados.
- [x] `npm.cmd ci` concluído fora do sandbox: 104 pacotes instalados; `npm ls --depth=0` sem dependências ausentes. A primeira tentativa falhou por bloqueio de rede (`EACCES`).
- [x] Corrigidas as 7 vulnerabilidades npm por atualizações semver compatíveis; `npm audit fix` reportou zero vulnerabilidades. O `npm ci`, `npm ls --depth=0` e build Vite passaram após a atualização.
- [x] Renomeado `desktop/vite.config.js` para `desktop/vite.config.mjs` para remover o aviso de loader CJS/ESM do Vite 8.
- [x] Desktop validado em snapshot limpo: `npm ci`, `npm ls --depth=0` e build passaram; apesar do aviso sobre postinstall não aprovado, `ffmpeg.exe` está presente. Licença do binário continua pendente para distribuição.
- [x] Escolhida a rota inicial de reprodução por métricas congeladas; cópia temporária mínima reproduziu contagens e CSVs sem vídeos, pesos, embeddings ou nova inferência. Leitura dos manifests corrigida para BOM UTF-8; diferenças de ponto flutuante ficaram abaixo de 1e-12.
- [x] Snapshot temporário da árvore-fonte validado: `npm ci`, `npm ls --depth=0`, audit npm sem vulnerabilidades, build Vite e reprodução das métricas passaram. O snapshot exclui assets locais do artigo e não contém dados/pesos não versionados.
- [x] Criada auditoria preliminar de licenças; confirmadas pendências do código autoral, dados/derivados, Ultralytics e ffmpeg-static. Ultralytics movido para o grupo Python opcional de experimentos.
- [x] Inventario revisado: relatórios 94-106 e matriz E2E classificados como registros do projeto; seleção do subconjunto de distribuição permanece condicionada à revisão de direitos.
- [ ] Instalar os grupos opcionais e a receita base em cópia limpa; isso ainda não foi feito.
- [ ] Revisar autorizações/licenças, finalizar manifesto e criar snapshot/commit local após staging seletivo.
