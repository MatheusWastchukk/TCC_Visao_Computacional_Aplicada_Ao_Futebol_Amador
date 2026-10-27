# Relatório 105 — Direcionamento Final do TCC e Plano de Entrega

**Data de referência:** 25/10/2026<br>
**Prazo previsto:** 23/11/2026 (**29 dias corridos após 25/10**)<br>
**Projeto:** Detecção e Análise de Eventos em Vídeos de Futebol Utilizando Técnicas de Visão Computacional<br>
**Natureza:** auditoria de situação e plano de execução; **não** constitui novo resultado experimental ou alteração no freeze.


> **Escopo desta frente local (25/10/2026):** o artigo está sendo desenvolvido no Overleaf. Para os ajustes do repositório, ficam fora de escopo a fonte LaTeX, o PDF, template, figuras e qualquer etapa de compilação do artigo. O checkout local serve para código, ambiente e artefatos de dados/resultados necessários ao trabalho. Esta delimitação do usuário substitui recomendações deste relatório que peçam recuperar ou empacotar arquivos do artigo.
## 1. Objetivo e decisão executiva

Os experimentos algorítmicos principais e a avaliação final das políticas experimentais V3 já foram concluídos. A prioridade passa a ser **(1) revisão acadêmica com o orientador, (2) integridade e reprodutibilidade do pacote de entrega e (3) preparação da versão final e da defesa**. Novos modelos, ajustes de ranking e reabertura de TEST não devem fazer parte desta etapa.

A investigação opcional sobre **tempo humano de revisão** pode contribuir com o artigo, mas só terá início após a confirmação do orientador e dos procedimentos éticos/institucionais aplicáveis, e jamais às custas da entrega em 23/11.

**Decisão recomendada:** entregar primeiro ao orientador o artigo atual e iniciar a consolidação do repositório e dos artefatos; só depois decidir sobre o piloto humano.

---

## 2. Documentos considerados e evidências

1. `TCC_Matheus_Wastchuk (1).pdf` — versão atual do artigo, 23 páginas, com pergunta de pesquisa, resultados de TEST, síntese de protocolos e confirmação dos E2E em 25/10.
2. `104_plano_piloto_revisao_humana_train.md` — estudo piloto condicional, restrito a TRAIN, com decisão GO/NO-GO prevista até 31/10.
3. `requisitos.md` — matriz de critérios experimentais, funcionais e de reprodutibilidade; APP-06 confirma os retestes de undo e cancelamento em 25/10.
4. `entrega_reprodutivel.md` — inventário de entrega, manifesto preliminar e pendências de pacote final.
5. Parecer parcial do orientador e relatórios 97–103 — contexto para as prioridades, especialmente avaliação TEST, ganho operacional, validação da aplicação e apresentação dos resultados.

Os fatos abaixo derivam desses materiais; as prioridades, prazos intermediários e critérios adicionais de aceite são **recomendações de planejamento**, não novas observações experimentais.

---

## 3. Estado atual: o que está demonstrado

### 3.1. Pesquisa e validação

O artigo informa um corpus de **42 partidas, 61h41min35s e 264 gols catalogados**, com durações calculadas sobre os vídeos efetivamente usados:

| Conjunto | Partidas | Duração dos arquivos | Gols catalogados |
|---|---:|---:|---:|
| TRAIN | 25 | 38h00min04s | 156 (152 avaliáveis no protocolo relevante) |
| VALIDATION | 8 | 11h53min43s | 51 |
| TEST | 9 | 11h47min48s | 57 |
| **Total** | **42** | **61h41min35s** | **264** |

A auditoria do conjunto amplo de candidatos identificou janelas relacionadas a **149/152 gols avaliáveis em TRAIN** e **50/51 em VALIDATION**, mas esses valores representam o *pool* antes da seleção, e não a cobertura da lista efetivamente entregue.

A avaliação **Nested LOMO em 25 partidas TRAIN** da família `NO_RESCUE` estimou a cobertura do procedimento de escolha sob orçamentos: **105/152** com teto de 30 minutos, **125/152** com 40 minutos e **132/152** com 58 minutos. Trata-se de estimativa fora do fold do **procedimento**, não da configuração V3 concreta usada em TEST.

### 3.2. Resultados finais das políticas V3 em TEST

O artigo informa que os 57 instantes de gol foram anotados por um operador a partir dos vídeos originais antes da inferência; políticas e modelos estavam congelados.

| Política experimental | Cobertura de TEST | Minutos únicos médios/partida | Teto | Jogos acima do teto |
|---|---:|---:|---:|---:|
| `FAST_V3` | 45/57 — **78,95%** | **27,27 min** | 30 min | 0/9 |
| `BALANCED_V3` | 50/57 — **87,72%** | **36,34 min** | 40 min | 0/9 |
| `HIGH_RECALL_V3` | 54/57 — **94,74%** | **49,60 min** | 58 min | 0/9 |

O resultado principal é **cobertura temporal de timestamps do catálogo em janelas selecionadas**. **Não** é precisão de classificação automática, validação da identidade da bola em toda a partida ou economia medida de tempo humano. O executor **V3 experimental é distinto do fluxo V4 da interface Electron**; não afirmar que a aplicação oferece os três modos V3 sem implementação e teste específicos.

### 3.3. Volume de vídeo disponibilizado

Com base na duração total de TEST de **11h47min48s = 707,8 minutos** e nas médias de nove partidas reportadas no artigo, a razão agregada *minutos selecionados totais / minutos originais totais*, calculada a partir das médias arredondadas, é aproximadamente:

| Política | Vídeo disponibilizado sobre duração total de TEST | Redução correspondente de material disponibilizado |
|---|---:|---:|
| FAST_V3 | **34,68%** | **65,32%** |
| BALANCED_V3 | **46,21%** | **53,79%** |
| HIGH_RECALL_V3 | **63,07%** | **36,93%** |

Esses percentuais correspondem à **razão de totais**, conforme a duração atualizada dos vídeos de TEST. Se forem apresentados também percentuais médios *por partida*, eles devem receber outro rótulo, pois não necessariamente coincidem. **Nenhuma dessas reduções equivale a tempo humano economizado.** Conferir os valores de precisão total nos CSVs originais para qualquer arredondamento final.

### 3.4. Aplicação e testes

O aplicativo V4 atual é Electron/React + FastAPI/FFmpeg. O artigo atualizado e `requisitos.md` registram **13/13 cenários E2E manuais aprovados**, abrangendo reprodução e navegação, classificação, notas, persistência e retomada de sessões, exportação, rastreabilidade e tratamento de erro. Em **25/10**, foram retestados e confirmados **desfazer a última decisão** e **cancelar análise em execução**.

Isso sustenta a apresentação do produto como **protótipo funcional verificado nos cenários exercitados**. Não sustenta, isoladamente, usabilidade geral, executável portátil, funcionamento garantido em outras máquinas, medição de latência do cancelamento ou economia de tempo de revisão.

---

## 4. Avaliação das pendências após o parecer do orientador

| Recomendação do parecer | Situação em 25/10 | Ação residual |
|---|---|---|
| Finalizar avaliação independente de TEST | **Concluída** | Preservar hashes, resultados e limitações; não reabrir para escolher novas políticas |
| Evidenciar ganho operacional | **Atendida para volume de vídeo; não para tempo humano** | Manter tabela de cobertura × fração de vídeo; decidir se piloto é viável |
| Distinguir protocolos e métricas | **Atendida no texto** | Verificar legibilidade da Tabela 3 e coerência de títulos/siglas |
| Explicar limites de BALL e tracking | **Documentada** | Revisar clareza sem alterar métricas ou denominadores |
| Explicar cobertura × custo | **Documentada** | Priorizar visualização simples e interpretação honesta em apresentação |
| Registrar ameaças à validade | **Documentada** | Preservar ressalvas: predomínio de certas câmeras, VAL reutilizada, 2 jogos TEST historicamente expostos, catálogo e um único anotador TEST |
| Validar fluxo da aplicação | **13/13 E2E manuais conforme documentação atual** | Preservar matriz, evidências e limites; não chamar de avaliação formal de usabilidade |
| Revisar apresentação/síntese final | **Em andamento** | Melhorar diagramação, compilar artefatos finais e obter feedback do orientador |

**Leitura geral:** não há justificativa, a partir das recomendações já recebidas, para treinar outro YOLO, iniciar R3D ou alterar os rankings. O esforço mais útil está agora na **qualidade editorial e na rastreabilidade**.

---

## 5. Frente P0-A — revisão acadêmica e envio ao orientador

**Janela sugerida: 25–28/10/2026.**

### Passo a passo

1. Partir do PDF atualizado de 23 páginas, não de um rascunho anterior com resultados V2 pendentes.
2. Fazer leitura final do resumo/Abstract, pergunta de pesquisa, metodologia, resultados TEST, discussão e conclusão, buscando contradições entre texto, tabelas e figuras.
3. Melhorar a **Tabela 3 de síntese dos protocolos**, na página 12: ela está densa, com colunas estreitas e quebras prejudiciais à leitura. Priorizar largura, abreviações explicadas e posicionamento de figura/tabela; evitar um segundo quadro que repita a mesma informação.
4. Conferir se todas as figuras inseridas são legíveis, pertencem às experiências descritas e estão citadas no texto; revisar quebras de página e espaços excessivos.
5. Confirmar que `V3`, `V4`, `LOMO`, `Nested`, `RRF`, `NMS`, `Candidate Pool`, `oracle` e `POSSIBLE_GOAL` são definidos antes do uso significativo.
6. Enviar a versão ao professor Neilor e solicitar foco nas conclusões pós-TEST, clareza da contribuição e eventuais exigências de entrega/banca.
7. Registrar a versão enviada, data, arquivo e feedback recebido em um único registro de revisão.

### Aceite

- [ ] PDF abre e apresenta todas as 23 páginas sem erros de renderização relevantes.
- [ ] A Tabela 3 é legível e referências cruzadas/legendas estão corretas.
- [ ] Resumo, resultados e conclusão usam os mesmos denominadores e ressalvas.
- [ ] Uma versão identificada foi enviada ao orientador para avaliação final.

**Importante:** aguardar feedback não bloqueia a frente paralela de reprodutibilidade.

---

## 6. Frente P0-B — pacote reproduzível e integridade de entrega

**Janela sugerida: 29/10–04/11/2026**, podendo começar em paralelo com a revisão do artigo.

O documento `entrega_reprodutivel.md` registra que o pacote atual ainda é preliminar. Ele distingue adequadamente dois níveis:

- **Reproduzir os números publicados:** usar CSVs/JSONs congelados, scripts de agregação, hashes e relatório, sem inferir novamente os vídeos de TEST.
- **Reexecutar inferência completa:** exige vídeos e pesos autorizados, ambiente compatível, FFmpeg, dependências e hardware apropriado; não significa autorização para retuning.

### Passo a passo

1. **Consolidar a versão do código:** revisar os arquivos locais modificados/não rastreados, separar o que pertence ao trabalho final do que é temporário, criar commit/snapshot identificável e registrar SHA completo.
2. **Recuperar todos os recursos do artigo:** fonte LaTeX, `article-config/` correspondente à compilação atual, bibliografia, figuras referenciadas, estilos/template e PDF final. O inventário anterior identificou ausência de `article-config/` no checkout consultado; confirmar onde está a versão legítima.
3. **Organizar as evidências experimentais mínimas:** freeze V3; hashes dos modelos; catálogo humano TEST congelado; protocolo e relatório TEST; `summary_test_v1.json`, `per_match_test_v1.csv`, `paired_mode_gains_test_v1.csv`, `goal_coverage_test_v1.csv`; avaliação Nested e scripts de agregação.
4. **Fixar instruções do ambiente real:** Python 3.11, dependências, Node/npm, FFmpeg, PyTorch/ROCm, caminhos de configuração e requisitos de GPU/CPU, com distinção explícita entre o que foi testado e o que não foi.
5. **Auditar licenças e distribuição:** não incluir por padrão vídeos brutos, clipes, pesos de terceiros, dados pessoais, ambientes virtuais e caches; documentar o motivo e requisitos de acesso a itens externos.
6. **Não reutilizar o release legado:** `release/GoalEventReview/` é PyInstaller/PySide6 de setembro; a aplicação validada é Electron/React + FastAPI. Um build Vite não equivale a instalador autocontido. Não prometer versão portátil sem a ter validado.
7. **Executar uma conferência limpa:** copiar o pacote para pasta separada e seguir as instruções de reprodução dos números; conferir caminhos, dependências de scripts, tabelas e arredondamentos; em outro teste opcional, iniciar o V4 com vídeo TRAIN autorizado.
8. **Emitir manifesto final:** paths, tamanho, SHA-256, estado de inclusão/exclusão, commit, versão de ambiente e instruções de reprodução.

### Aceite

- [ ] Git/snapshot da entrega identificado, sem depender de modificações locais não registradas.
- [ ] Fonte do artigo e todos os recursos necessários à sua compilação estão presentes e coerentes.
- [ ] As tabelas principais podem ser recalculadas a partir dos artefatos congelados **sem alterar TEST**.
- [ ] O manifesto final lista arquivos, hashes e exclusões autorizadas.
- [ ] O README não promete instalador, plataforma ou desempenho não testados.

---

## 7. Frente P1 — decisão condicional sobre piloto humano

**Gate sugerido: até 31/10/2026.** A proposta existente é adequada como plano **exploratório**, mas não deve ser tratada como obrigação do trabalho nem como validação da política V3.

### Condições para GO

- Orientador confirma utilidade acadêmica neste estágio.
- Procedimento ético/institucional aplicável é verificado **antes** de convidar participantes ou coletar dados.
- Há de três a cinco voluntários e partidas TRAIN apropriadas, com consentimento e privacidade assegurados.
- Os materiais, sessões e análise podem ser concluídos **sem atrasar a revisão do artigo, o pacote ou a entrega**.

### Desenho resumido, somente com GO

- Condição A: reprodução manual/cronológica da partida TRAIN.
- Condição B: revisão assistida pela aplicação **V4** em uma partida TRAIN diferente de dificuldade aproximada.
- Alternar a ordem A/B entre os participantes e não mostrar os timestamps de catálogo aos revisores.
- Medir tempo humano **ativo**, pausas, eventos encontrados/não encontrados, marcações sem correspondência ao catálogo, revisitas e observações de uso; manter tempo de processamento automático separado.
- Analisar resultados por participante, descritivamente; **3–5 pessoas não autorizam alegação geral de economia ou teste confirmatório**.

### NO-GO aceitável

Se faltar autorização, participantes ou prazo, registrar que **a economia de tempo humano não foi medida** e manter o estudo como trabalho futuro. Não coletar dados de participantes sem confirmar antes as exigências institucionais.

---

## 8. Frente P0-C — conclusão editorial e defesa

**Janela sugerida: 05–22/11/2026.**

1. Incorporar o feedback do orientador, preservando valores e protocolos dos resultados congelados.
2. Revisar todas as referências bibliográficas e garantir correspondência entre citações e bibliografia.
3. Conferir títulos, captions, tabelas, fontes de imagens, paginação e extensão exigida pela instituição.
4. Preparar apresentação: problema, objetivo, pipeline, dataset/split, problema BALL, pool vs shortlist, Nested, resultados TEST e conclusão/limitações.
5. Preparar demonstração segura do aplicativo com **vídeo TRAIN autorizado e sessão pré-computada**, sem depender de execução de inferência demorada durante a banca.
6. Preparar resposta para a questão central do parecer: por que priorizar seleção/revisão e como demonstrar em partidas novas — citar Nested e TEST, separando cobertura de tempo humano.
7. Gerar PDF final e pacote final, com os hashes/versões auditados; submeter com margem antes de 23/11 quando possível.

### Questões que a banca provavelmente explorará

- Por que 98% de cobertura no pool não significam 98% de gols recuperados na shortlist?
- Por que o modo que seleciona mais vídeo recupera mais eventos, mas não todos?
- Qual a diferença entre Nested OOF TRAIN, LOMO usada para seleção de V3 e TEST?
- Por que o sistema não confirma automaticamente um gol?
- Por que BALL/tracking não se tornaram um detector de evento confiável?
- A ferramenta realmente economiza tempo humano? **Ainda não há essa medida**.
- Por que V3 do experimento e V4 da aplicação são diferentes?

---

## 9. Cronograma de execução

| Intervalo | Atividade principal | Entrega verificável | Prioridade |
|---|---|---|---|
| **25–28/10** | Revisão visual/textual do PDF; primeira rodada com orientador | PDF enviado, Tabela 3 revisada e lista de observações | **P0** |
| **29/10–04/11** | Repositório e entrega reproduzível; teste de reprodução de métricas | Snapshot, artefatos necessários, hashes e README verificados | **P0** |
| **Até 31/10** | Gate do piloto humano | GO autorizado ou NO-GO documentado | **P1** |
| **05–14/11** | Incorporar feedback e, condicionalmente, coletar piloto TRAIN | Manuscrito revisado e eventual resultado piloto descritivo | **P0/P1** |
| **15–18/11** | Consolidação de artigo, demonstração e apresentação | Versão quase final, slides e pacote de apoio | **P0** |
| **19–22/11** | Auditoria final e submissão com margem | PDF final e pacote reproduzível fechados | **P0** |
| **23/11** | Prazo previsto | Entrega acadêmica | **Obrigatório** |

As datas intermediárias são sugestões de gestão; somente o prazo de entrega indicado foi tratado como referência institucional fornecida ao projeto.

---

## 10. Matriz de riscos e decisões

| Risco | Consequência | Controle recomendado |
|---|---|---|
| Artigo e checkout divergentes | Figuras/compilação irreproduzíveis | Fixar SHA/snapshot e incluir `article-config`, figuras e fontes exatos |
| Exigir instalador portátil no fim | Desviar tempo de entrega | Confirmar exigência; caso não exista, entregar código + reprodução documentada |
| Estudo humano sem autorização/tempo | Risco ético e atraso | Gate GO/NO-GO até 31/10 e estudo restrito a TRAIN |
| Confundir V3 com V4 | Alegar que a interface faz algo não avaliado | Separar executor experimental e aplicação operacional em texto e demonstração |
| Métrica operacional mal interpretada | Alegar economia de tempo humano não medida | Usar “minutos únicos de vídeo disponibilizado” e informar fração agregada |
| Novos ajustes após TEST | Comprometer interpretação da avaliação | Manter modelos/políticas TEST congelados e não usar TEST para desenvolvimento |
| Ausência de feedback do orientador | Pouco tempo para correções finais | Enviar a versão atual imediatamente e trabalhar em paralelo na reprodutibilidade |

---

## 11. Checklist executivo de fechamento

### Ciência / relatório

- [x] Nested LOMO 25/25 documentado e distinto de seleção LOMO dos modos V3.
- [x] Avaliação final TEST V3 executada e relatada, sem retuning conforme documentos fornecidos.
- [x] Limitações BALL/tracking, cobertura temporal e ausência de tempo humano explicitadas.
- [x] Tabela-síntese de protocolos incorporada ao artigo.
- [ ] Conferir tabelas, legendas, layout e referências na versão final compilada.
- [ ] Revisão final do orientador sobre resultados e conclusão.

### Aplicação

- [x] 13/13 E2E manuais documentados no artigo atualizado e requisitos.
- [x] Undo e cancelamento retestados em 25/10; latência não medida.
- [ ] Preservar provas/matriz dos E2E no pacote de entrega.

### Reprodutibilidade

- [ ] Confirmar commit/snapshot e consolidar arquivos modificados.
- [ ] Incluir todos os recursos LaTeX e figuras necessários.
- [ ] Conferir CSVs e manifests congelados contra números publicados.
- [ ] Confirmar autorizações e exclusões de pesos/vídeos.
- [ ] Emitir e testar o manifesto/pacote final.

### Decisão opcional

- [ ] Orientador/instituição decidiram GO ou NO-GO do piloto humano até 31/10.
- [ ] Se GO, conduzir somente em TRAIN, descritivamente e sem comprometer a entrega; se NO-GO, registrar trabalho futuro.

### Entrega

- [ ] Slides e demonstração segura com material autorizado.
- [ ] PDF final e artefatos revisados e enviados dentro do prazo.

---

## 12. Conclusão e orientação de trabalho

O projeto já demonstrou, em nove partidas TEST, que políticas congeladas selecionam trechos com **78,95% a 94,74% de cobertura temporal dos gols catalogados**, dependendo da quantidade de vídeo disponibilizada. O estudo também documenta a alta cobertura do pool, a dificuldade de escolher candidatos sob orçamento e as limitações de BALL/tracking. A interface V4 foi exercitada em testes manuais, mas não houve prova de economia de tempo humano.

**Próxima ação concreta em 25/10:** enviar a versão atual de 23 páginas ao professor Neilor para revisão final **e**, em paralelo, abrir a auditoria de reprodutibilidade com foco em Git, `article-config`, figuras, CSVs, hashes e README. Até 31/10, obter a decisão condicional sobre o piloto de revisão humana. Não iniciar novos experimentos de IA nem reabrir TEST.

O objetivo restante não é ampliar a pesquisa, mas **entregar o trabalho existente com clareza, verificabilidade e qualidade acadêmica**.
