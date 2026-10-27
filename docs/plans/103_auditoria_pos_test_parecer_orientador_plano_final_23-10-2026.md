# Relatório 103 — Auditoria pós-TEST, atendimento ao parecer do orientador e plano final do TCC

**Data de referência:** 23/10/2026<br>
**Atualização da aplicação:** 25/10/2026
**Trabalho:** *Detecção e Análise de Eventos em Vídeos de Futebol Utilizando Técnicas de Visão Computacional*<br>
**Prazo previsto para entrega:** 23/11/2026<br>
**Objetivo:** consolidar o que foi efetivamente executado depois do Plano 100; confrontar os resultados com as oito recomendações do orientador; registrar verificações ainda pendentes; e definir as últimas tarefas para a versão final do artigo.

> **Natureza deste documento:** auditoria e plano de conclusão, não novo experimento nem novo manifesto de freeze. O TEST já foi avaliado e **não deve ser reaberto, usado para retuning ou repetido para escolher uma política**. Em caso de divergência narrativa, prevalecem os artefatos congelados conferidos e identificados por hash neste relatório.

---

## 1. Resumo executivo

A maioria das pendências prioritárias do parecer foi atendida. O ponto mais importante, a **avaliação final das políticas congeladas no split TEST**, foi concluído nas nove partidas e em 57 timestamps humanos anotados às cegas para as predições. Duas partidas tiveram exposição histórica anterior no projeto, limitação discutida no artigo. Os resultados são:

| Modo experimental V3 | Gols cobertos em TEST | Cobertura | Vídeo único médio por partida | Teto nominal | Jogos acima do teto |
|---|---:|---:|---:|---:|---:|
| `FAST_V3` | 45/57 | **78,95%** | 27,27 min | 30 min | 0/9 |
| `BALANCED_V3` | 50/57 | **87,72%** | 36,34 min | 40 min | 0/9 |
| `HIGH_RECALL_V3` | 54/57 | **94,74%** | 49,60 min | 58 min | 0/9 |

Esses valores representam **cobertura temporal de timestamps de gols catalogados**: um evento é coberto se o instante humano pertence a pelo menos uma janela selecionada. **Não** são precisão de detecção automática, acurácia de reconhecimento de gol, taxa de falsos positivos nem tempo humano economizado.

O artigo-fonte LaTeX foi atualizado em 23/10 com resultados de TEST, descrição das políticas, limitações e conclusão. Em 25/10, os dois recursos antes pendentes foram implementados e testados manualmente pelo usuário na aplicação: desfazer a última decisão e cancelar uma análise em andamento; ambos funcionaram segundo a confirmação recebida. A compilação Vite e as verificações funcionais focadas do backend também passaram. A latência do cancelamento não foi medida e a porcentagem de uso da GPU segue indisponível neste ambiente.

**Conclusão executiva:** não há justificativa para abrir novos treinamentos, heurísticas de ranking, rastreadores ou R3D nesta fase. As duas verificações numéricas/editoriais foram concluídas diretamente nos artefatos. O trabalho restante concentra-se na compilação e revisão do artigo, rastreabilidade final e eventual avaliação-piloto com revisores, somente se houver prazo e condições institucionais.

---

## 2. Documentos utilizados e hierarquia de evidência

1. `Avaliação TCC Matheus Wastchuk(1).docx`: parecer do orientador, com oito recomendações e a pergunta sugerida para a banca.
2. `101_auditoria_pre_test_v3_17-10-2026.md`: auditoria de integridade do freeze V3 e a deficiência inicialmente encontrada no executor V3, posteriormente resolvida para avaliação final.
3. `102_protocolo_anotacao_cega_timestamps_test_v3_17-10-2026.md`: cegamento, convenção dos timestamps, validação e freeze do catálogo humano.
4. `102_relatorio_atualizacao_artigo_resultados_test_v3_20-10-2026.md`: síntese de TEST, E2E e alterações aplicadas no manuscrito.
5. `application_e2e_matrix_2026-10-20.csv`: **fonte primária disponível para contagem e status dos 13 cenários E2E**.
6. `docs/TCC_Matheus_Wastchuk.tex`: fonte LaTeX atual do artigo, atualizado com as seções de TEST e conclusão.
7. Histórico metodológico: Nested LOMO 25/25, Relatórios 97–99, freeze V3 e Plano 100.

**Para resolver divergências**, priorizar os artefatos detalhados de `TEST-V3-FINAL-EVALUATION-20261019` (`goal_coverage_test_v1.csv`, `per_match_test_v1.csv`, `paired_mode_gains_test_v1.csv`, `summary_test_v1.json`) sobre descrições narrativas em relatórios. Esses arquivos estão disponíveis neste checkout e foram conferidos por hash e conteúdo na seção 5.

---

## 3. O que o resultado final realmente demonstra

### 3.1. População e protocolo

- Corpus integral: **42 partidas**, **61h41min35s** e **264 gols catalogados**, conforme a soma das durações dos arquivos em TRAIN, VALIDATION e TEST informadas no artigo.
- TRAIN: **25 partidas**, 156 gols do catálogo-base, sendo **152 avaliáveis** no protocolo principal; o Nested utilizou **5.742 grupos** e 25 outer folds.
- VALIDATION: **8 partidas**, 51 gols; histórico de consultas durante o desenvolvimento.
- TEST: **9 partidas**, **11h47min48s de vídeo efetivamente analisado**, **57 timestamps humanos** anotados a partir dos vídeos originais sem visualização das predições, com catálogo congelado antes da inferência.
- Políticas V3 fixas: todas usam `RRF4_NET`, família `NO_RESCUE` e gap 45 s; o modo rápido usa 30 s/K60, o balanceado 40 s/K60 e o de maior cobertura 60 s/K55.
- A avaliação V3 foi feita por **executor experimental distinto do fluxo operacional V4 da interface**. A interface existente não deve ser descrita como se já oferecesse os três modos V3.

### 3.2. Interpretação correta das três camadas de resultado

| Camada | Conjunto/protocolo | Exemplo | Interpretação válida |
|---|---|---|---|
| Candidate Pool | TRAIN, todas as janelas | 149/152 = 98,03% | Teto de cobertura da mineração antes de ranquear/cortar |
| Candidate Pool | VALIDATION, todas as janelas | 50/51 = 98,04% | Teto de pool naquele split consultado anteriormente |
| V4 operacional | VALIDATION, modo `alta_cobertura` | 47/51; 55,98 min/jogo | Cobertura do modo V4, **não** dos modos V3 |
| Nested LOMO OOF | TRAIN 25/25, família NO_RESCUE, orçamento 30/40/58 min | 105/152; 125/152; 132/152 | Estimativa de desempenho do **procedimento de seleção por orçamento** |
| LOMO de seleção de configurações | TRAIN 25/25, 72 configurações comparadas | 111/152; 126/152; 133/152 | Critério usado para fixar V3; **seleção otimista**, não validação final |
| TEST final | 9 partidas, políticas V3 fixas | 45/57; 50/57; 54/57 | Cobertura observada de políticas congeladas em partidas reservadas, com ressalvas de independência histórica |

A alta cobertura do pool e a menor cobertura das shortlists **não** são medições contraditórias: uma pergunta se o candidato existe; a outra, se foi selecionado dentro do orçamento.

### 3.3. Cobertura dos três modos não é perfeitamente aninhada

Em TEST, a cobertura agregada sobe de 45 para 50 e depois para 54 eventos. Entretanto, aumentar a janela ou o orçamento **não significa** necessariamente preservar cada evento do modo anterior, pois as políticas fixas variam K e seleção.

Os CSVs pareados confirmam que BALANCED cobriu cinco eventos adicionais em relação a FAST, sem exclusivos de FAST nessa comparação; HIGH cobriu dez eventos não cobertos por FAST e perdeu um evento que FAST cobria; comparado a BALANCED, HIGH cobriu cinco adicionais e perdeu um. Os números do artigo e do Relatório 102 estão alinhados aos CSVs.

### 3.4. O significado do resultado em TEST

É legítimo concluir que as políticas V3 selecionaram, nas nove partidas avaliadas, um subconjunto de vídeo que continha entre **78,95% e 94,74% dos timestamps catalogados**, respeitando seus tetos de duração em todos os jogos observados.

**Não é legítimo concluir** que o modelo detecta automaticamente 94,74% dos gols com precisão equivalente, que reduziu em determinada porcentagem o tempo de trabalho do revisor, ou que mantém necessariamente o mesmo desempenho em ligas/câmeras diferentes.

---

## 4. Atendimento às oito recomendações do orientador

| Nº | Recomendação original | Situação em 23/10 | Evidência e decisão |
|---:|---|---|---|
| **1** | Concluir avaliação final das políticas congeladas em TEST | **Concluída** | 9 partidas, 57 instantes anotados às cegas para as predições, três políticas fixas avaliadas; duas partidas tiveram exposição histórica anterior; não executar novo TEST. |
| **2** | Evidenciar o ganho operacional real | **Parcial** | Frações agregadas de vídeo conferidas e corretamente definidas; tempo humano não medido. Piloto humano facultativo. |
| **3** | Tornar protocolos e resultados fáceis de comparar | **Quase concluída** | Manuscrito distingue TRAIN Nested, seleção TRAIN, V3 TEST e app V4; tabela-síntese de protocolos adicionada, pendente de revisão no PDF. |
| **4** | Explicar limites de detecção/identidade BALL | **Atendida, com revisão de linguagem** | O artigo já contém 80/298 propostas compatíveis e 110/359 identidades corretas em populações diagnósticas distintas; explicitar consequência para candidatos. |
| **5** | Reforçar compromisso cobertura × vídeo | **Atendida em dados; revisar apresentação** | Tabela e curva Nested, tabela TEST e minutos existem; frações de duração estão definidas como razão agregada e aguardam inspeção visual no PDF final. |
| **6** | Analisar vieses e validade externa | **Atendida em conteúdo** | Incluir em um parágrafo bem identificado domínio Veo, VALIDATION reutilizada, duas partidas TEST historicamente expostas, catálogo e anotador único. |
| **7** | Validar aplicação ponta a ponta | **Concluída com ressalva de medição** | Os 13 cenários estão registrados como aprovados; E2E-06 e E2E-07 foram retestados manualmente pelo usuário em 25/10. Sem estudo formal de usabilidade ou medição de latência do cancelamento. |
| **8** | Revisar a apresentação, termos e conclusão | **Pendente de revisão final** | Compilar LaTeX, inspecionar figuras/tabelas/referências, uniformizar nomenclatura, obter nova leitura do orientador e fechar o PDF para entrega. |

**Leitura geral do parecer:** a principal pendência científica inicialmente levantada, TEST, foi resolvida. O foco agora é melhorar comunicação, qualidade editorial, replicabilidade da evidência e honestidade das limitações.

---

## 5. Auditorias numéricas/editoriais concluídas

As duas verificações foram realizadas diretamente nos artefatos da tentativa final válida, sem reprocessar vídeos nem alterar a avaliação TEST. Os CSVs citados no Relatório 102 estão disponíveis no checkout.

### 5.1. Fração da duração original selecionada

O conjunto TEST processado soma **707,797 minutos (11h47min48s)**, conforme summary_test_v1.json e a soma das durações em per_match_test_v1.csv. A fração publicada no artigo é a razão entre os minutos únicos totais selecionados e os minutos originais totais:

| Modo | Minutos únicos totais | Duração original total | Fração selecionada | Redução de duração |
|---|---:|---:|---:|---:|
| FAST_V3 | 245,460 | 707,797 | **34,68%** | **65,32%** |
| BALANCED_V3 | 327,088 | 707,797 | **46,21%** | **53,79%** |
| HIGH_RECALL_V3 | 446,411 | 707,797 | **63,07%** | **36,93%** |

Fórmula: 100 × soma(T_único_i) / soma(T_original_i). Os percentuais de 34,68%, 46,21% e 63,07% estão corretos e representam a razão agregada do corpus. Não são a média simples dos percentuais por partida nem uma medição de tempo humano economizado.

O valor de **12h24min51s (744,85 min)** usado em uma versão anterior deste relatório não corresponde à duração efetivamente analisada em TEST. Ele foi substituído pelo total registrado nos artefatos canônicos e na tabela do corpus do artigo.

### 5.2. Ganhos e perdas pareados entre políticas

paired_mode_gains_test_v1.csv confirma os números do artigo:

| Comparação | Cobertos por ambos | Só política A | Só política B | Por nenhum |
|---|---:|---:|---:|---:|
| FAST_V3 × BALANCED_V3 | 45 | 0 | 5 | 7 |
| FAST_V3 × HIGH_RECALL_V3 | 44 | 1 | 10 | 2 |
| BALANCED_V3 × HIGH_RECALL_V3 | 49 | 1 | 5 | 2 |

Assim, a frase anterior do Relatório 102 que atribuía um evento exclusivo de FAST a cada modo mais longo estava incorreta para a comparação FAST × BALANCED. O artigo e o Relatório 102 atualizado descrevem corretamente os pares.

### 5.3. Hashes dos artefatos conferidos

- summary_test_v1.json: A1213514E263FB0972BDB31726F1B0F4FFDA6A2F00533B058B16B1DC69377033
- per_match_test_v1.csv: B184FBE616B1E6A19CE0DCFD89BB39540CAB3299EC59023DABACA7E0D5703D15
- paired_mode_gains_test_v1.csv: 4E4B04ED0430755CAF9FDCFFB2E46F5B87B1A37C9701B369C687979DBC98D6BA
- goal_coverage_test_v1.csv: 263AAB525BCE579C07C69F18D76B10E85375780C11E45F18AA132DCC11E78ADA

As verificações foram documentais e determinísticas sobre os resultados congelados. Não houve alteração de escores, janelas, rótulos ou métricas primárias de TEST.

---

## 6. Auditoria da aplicação: matriz E2E de 13 cenários

A matriz preserva as evidências dos testes anteriores. Em 25/10, o usuário também retestou E2E-06 e E2E-07 na aplicação e confirmou que ambos funcionaram. Assim, não há pendência funcional nem reteste manual aberto para esses dois itens.

| Grupo de cenários | Estado atual | Evidência |
|---|---|---|
| E2E-01 a E2E-05 e E2E-08 a E2E-13 | Aprovados conforme os registros anteriores | Confirmações manuais e artefatos listados na matriz. |
| E2E-06: desfazer decisão | Aprovado no reteste manual de 25/10 | O usuário confirmou que o desfazer funcionou na aplicação. |
| E2E-07: cancelar análise | Aprovado no reteste manual de 25/10 | O usuário confirmou que o cancelamento funcionou na aplicação; a latência não foi cronometrada. |
| Situação consolidada | 13 cenários aprovados | O registro não constitui estudo formal de usabilidade. |

### 6.1. Funcionalidades demonstradas no registro

- Inicialização Electron/API após ajustar tempo de espera de 10 para 60 segundos (E2E-01).
- Player, seek, navegação, storyboard, velocidade e atalhos de velocidade após correção de foco (E2E-02).
- Rótulos GOAL/NO_GOAL/NOT_EVALUABLE, adiar e filtros (E2E-03).
- Persistência de notas/decisões, confirmação personalizada e edição de notas (E2E-04).
- Reabertura de sessão parcialmente revisada ou concluída (E2E-05).
- Desfazer a última classificação, restaurando rótulo e estado anteriores e mantendo histórico por sessão; reteste manual confirmado pelo usuário (E2E-06).
- Cancelar análise em andamento, interromper nos checkpoints, remover a saída parcial e manter a sessão identificada como cancelada; reteste manual confirmado pelo usuário (E2E-07).
- Tratamento de MP4 inválido, sem sucesso falso e com modal de erro útil (E2E-08).
- Compatibilidade com sessão antiga (E2E-09).
- Exportações HTML, CSV, JSON e MP4 (E2E-10).
- Identificação correta de modos operacionais V4, distintos das políticas experimentais V3 (E2E-11).
- Rastreabilidade da versão 0.4.0, hardware AMD e configurações em sessão (E2E-12).
- Análise integral de vídeo TRAIN concluída no pipeline V4 (E2E-13).

### 6.2. Limites da verificação

- **Medição do cancelamento:** o fluxo foi testado manualmente e confirmado pelo usuário; não foi cronometrada a latência nem caracterizada a resposta por etapa ou equipamento.
- **Telemetria:** a porcentagem de uso da GPU continua indisponível neste ambiente.
- **Usabilidade:** a aprovação dos cenários funcionais não constitui estudo formal de usabilidade nem demonstra economia de tempo humano.

Os dois itens deixam de ser pendências de implementação e de reteste manual. Os limites restantes são medições adicionais, sem impacto sobre o algoritmo ou os resultados de TEST.

---

## 7. Revisões específicas do artigo antes da próxima avaliação pelo orientador

### 7.1. Resumo e Abstract

Manter explícitos 42 jogos, 57 timestamps de TEST, três coberturas de TEST, minutos médios, papel da revisão humana e ausência de medição de precisão/tempo humano. Não substituir avaliação de TEST por números Nested de TRAIN.

### 7.2. Corpus e Metodologia

Conferir: 25/8/9 partidas; TRAIN com 152 gols avaliáveis no protocolo, TEST com 57; cegamento do anotador, freeze anterior à inferência, limite das exposições históricas e regra exata `start ≤ t ≤ end` sem tolerância pós-hoc. Distinguir V4 operacional e executor V3.

### 7.3. Resultados: tabela-síntese e gráfico único

A tabela-síntese de protocolos foi adicionada ao artigo. Manter a figura Nested 25/25 para demonstrar seleção sob orçamento e conferir no PDF se a tabela final de TEST comunica cobertura e minutos únicos de modo legível. Os percentuais de duração estão rotulados como razão agregada dos totais.

### 7.4. BALL, tracking e limitações

Explicar causalmente: bola pequena/ocluída → proposta ausente ou incorreta → associação instável → sinal espacial pouco confiável → necessidade de priorização e revisão humana. Identificar 80/298 como correspondência espacial de **frames selecionados** e 110/359 como identidade correta em **outro protocolo**; não apresentá-los como uma mesma taxa geral nem extrapolar para todas as partidas.

### 7.5. Discussão, validade e conclusão

Consolidar limitações: domínio de aquisição dominado por Veo e semelhantes; VALIDATION consultada historicamente; exposição histórica de duas partidas TEST; catálogo criado por operador único; falta de concordância interanotador, de precisão de evento e de tempo humano medido; protótipo V4 distinto do executor V3. A conclusão deve responder claramente o problema de pesquisa sem extrapolar além da cobertura dos timestamps.

### 7.6. Revisão visual do PDF LaTeX

Executar compilação local ou Overleaf do projeto real, conferir referências, tabelas, figuras, cortes de página, placeholders (`\imagemTCC`), legenda da imagem do app e ordem das seções. Verificar se a versão em PDF corresponde ao `.tex` modificado em 23/10; o PDF de 21 páginas anterior é historicamente anterior à incorporação de TEST. Revisar ortografia, primeira definição de siglas, referências comerciais e links bibliográficos. Não declarar compilação/revisão visual concluídas sem executá-las.

---

## 8. Estudo-piloto de tempo humano: possibilidade, não obrigação

O orientador pediu, **se possível**, um teste com revisores para observar o esforço humano de revisão. Os dados atuais não o medem.

**Proposta mínima**, dependente de viabilidade e orientação/aprovação institucional:

- Usar somente **partidas TRAIN**, sem usar TEST para desenvolver UX.
- Comparar revisão convencional de vídeo com revisão assistida pela aplicação em partidas diferentes mas comparáveis; alternar condição e ordem entre participantes.
- Recrutar idealmente 3–5 participantes, documentando experiência com futebol/vídeo e critérios de inclusão.
- Medir tempo humano ativo, eventos encontrados, falsos eventos reportados, revisitas e facilidade percebida.
- Não apresentar amostra pequena como prova estatística universal de economia; reportar observações descritivas e limitações.
- Separar claramente o tempo de processamento automático da duração de vídeo disponibilizado e do tempo humano de revisão.

**Gate:** realizar apenas se houver tempo para projeto, consentimento/procedimentos institucionais necessários e análise adequada. Caso contrário, registrar a avaliação humana como **trabalho futuro**. Não atrasar revisão editorial nem reabrir o TEST.

---

## 9. Plano de execução até a entrega de 23/11/2026

| Janela sugerida | Prioridade | Atividade | Entrega verificável |
|---|---|---|---|
| **23/10 — concluído** | P0 | Auditar definição dos percentuais e ganhos pareados nos artefatos existentes; inserir tabela-síntese de protocolos | Seção 5 deste relatório: fórmulas, contagens pareadas e hashes conferidos; tabela inserida no artigo |
| **27–31/10** | P0 | Compilar LaTeX, inspecionar PDF, verificar referências, tabelas/figuras e coerência de V3/V4 | PDF revisado visualmente, sem placeholders indevidos nem tabelas cortadas |
| **01–07/11** | P1 | Enviar ao orientador; incorporar ajustes redacionais; opcionalmente testar cancelamento seguro sem tocar no algoritmo | Versão para orientação + registro de respostas às oito recomendações |
| **08–14/11** | P1/condicional | Estudo-piloto humano somente se aprovado e viável; ou encerrar esse item como trabalho futuro | Breve protocolo e resultados descritivos, ou decisão documentada de não executar |
| **15–22/11** | P0 | Revisão final de texto, ABNT/template exigido, tabelas, coerência numérica, slides e versão de submissão | PDF definitivo, checklist de integridade e apresentação |
| **23/11** | Marco | Entrega prevista do TCC | Versão submetida conforme regras institucionais |

A prioridade absoluta é uma **versão correta e revisada**, não a criação de funcionalidades adicionais.

---

## 10. Checklist de conclusão

### Evidência científica e metodologia

- [x] Nested LOMO 25/25 e resultados TRAIN documentados.
- [x] Seleção e freeze V3 anteriores à inferência em TEST.
- [x] Anotação cega para as predições de 57 timestamps por um operador e catálogo fechado antes da inferência.
- [x] Uma avaliação preditiva final em TEST, com três políticas fixas e sem retuning relatado.
- [x] Resultados globais e tabela por partida de cobertura de TEST incorporados ao fonte atual.
- [x] Conferir os arquivos por evento e confirmar ganhos/perdas pareados narrados.
- [x] Conferir a fórmula das frações de vídeo; confirmar a razão agregada de duração.
- [ ] Arquivar/verificar referências a protocolo, freeze, modelos, hashes e relatórios finais em anexos ou repositório acessível à banca.

### Aplicação

- [x] 13 cenários E2E avaliados/documentados.
- [x] Persistência, retomada de sessão, classificação, exportações e execução TRAIN confirmadas nas evidências disponíveis.
- [x] Implementar undo persistente e cancelamento cooperativo; verificações funcionais focadas aprovadas.
- [x] Retestar E2E-06 e E2E-07 manualmente no Electron; usuário confirmou o funcionamento de ambos em 25/10/2026.

### Artigo e orientação

- [x] `.tex` atualizado com resumo, metodologia, TEST, discussão, limitações e conclusão.
- [x] Acrescentar tabela-síntese de protocolos ao manuscrito e registrar a resolução das duas verificações editoriais.
- [ ] Compilar o `.tex` atualizado e verificar figuras, paginação e referências.
- [ ] Enviar ao orientador e obter nova rodada de comentários.
- [ ] Preparar conclusão/defesa sem prometer detecção de gol plenamente automática ou economia humana não medida.
- [ ] Decidir formalmente se o piloto com revisores será realizado ou citado como trabalho futuro.

---

## 11. Resposta preparada para a pergunta da banca

**Pergunta do orientador:** se o pool já cobre quase todos os gols, mas a shortlist perde eventos e a bola é mal localizada, que melhoria priorizar e como demonstrar eficácia em partidas novas?

**Resposta tecnicamente defensável:**

> Priorizar a seleção de candidatos sob orçamento, combinada a uma ferramenta de revisão humana, em vez de depender imediatamente de um tracker de bola que ainda apresenta baixa confiabilidade. O pool contém janelas associadas à maior parte dos gols, mas elas precisam ser ordenadas e selecionadas sob um custo controlado. O Nested LOMO estimou esse trade-off durante o desenvolvimento e as três configurações congeladas foram avaliadas no split TEST, com cobertura de 78,95%, 87,72% e 94,74% dentro de seus respectivos tetos de vídeo. Duas partidas tiveram exposição histórica anterior no projeto. A comprovação de economia de tempo humano exigiria outro estudo, com revisores e medidas de esforço efetivo; por isso o trabalho não apresenta a duração selecionada como tempo humano poupado.

---

## 12. Decisão final

**O TCC não precisa de novos experimentos de IA para atender ao parecer recebido.** A recomendação 1 — a pendência mais importante — foi atendida. As duas verificações numéricas foram resolvidas a partir dos CSVs e incorporadas ao artigo e a este relatório. O caminho crítico agora é compilar e revisar o manuscrito atualizado, obter a leitura do orientador e finalizar a versão de entrega. O piloto humano permanece condicional e separado da evidência algorítmica.

Se o estudo de revisão humana não for realizado, a ausência de tempo humano medido deve permanecer explícita na Discussão e nos Trabalhos Futuros. Todos os aprimoramentos posteriores de interface devem permanecer separados da avaliação algorítmica V3 já congelada e concluída.

---

## Referências documentais internas

- `Avaliação TCC Matheus Wastchuk(1).docx` — parecer do orientador.
- `101_auditoria_pre_test_v3_17-10-2026.md` — preflight e preparação do executor V3.
- `102_protocolo_anotacao_cega_timestamps_test_v3_17-10-2026.md` — protocolo de timestamps humanos.
- `102_relatorio_atualizacao_artigo_resultados_test_v3_20-10-2026.md` — resultados finais, atualização do artigo e testes manuais.
- `application_e2e_matrix_2026-10-20.csv` — matriz fonte dos 13 cenários.
- `Texto colado(20261023-180659).txt` — manuscrito LaTeX atualizado em 23/10.
- `98_plano_selecao_modos_lomo_no_rescue_v3_14-10-2026.md`, `99_resultados_selecao_modos_lomo_no_rescue_v3_14-10-2026.md` — seleção pré-TEST dos modos.
- `97_relatorio_execucao_nested_25_train_freeze_v2_14-10-2026.md` — reconstrução full-pool e Nested 25/25.

**Nota de escopo da auditoria:** os resultados de TEST foram conferidos nos artefatos finais disponíveis, por conteúdo e hash. Esta atualização não reexecutou o pipeline, não abriu novamente os vídeos e não repetiu a inferência.
