# Relatório 102 — Atualização do artigo após a avaliação V3 em TEST

**Data:** 20/10/2026<br>
**Natureza:** relatório de resultados e registro da atualização aplicada ao manuscrito-fonte.<br>
**Base experimental:** avaliação final V3 em nove partidas TEST, descrita em [`test_v3_results_report.md`](../../experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/test_v3_results_report.md).<br>
**Situação do artigo no checkout:** o fonte atual [`TCC_Matheus_Wastchuk.tex`](../TCC_Matheus_Wastchuk.tex) foi fornecido e atualizado em 23/10/2026 com os resultados V3 TEST, distinção entre Nested LOMO TRAIN, seleção LOMO TRAIN e teste final, discussão, limitações e conclusão. A compilação LaTeX e a revisão do orientador permanecem pendentes.<br>
**Aplicação:** `npm.cmd run build --prefix desktop` passou em 20/10/2026 (Vite 8.3.0, 1.877 módulos). O Vite emitiu aviso sobre a configuração ESM/CJS; a compilação terminou com sucesso. A primeira tentativa interativa exibiu alerta da API; após ampliar a espera do Electron de 10 para 60 segundos, o usuário reiniciou a aplicação e confirmou que o alerta parou de aparecer. E2E-01 aprovado para inicialização e E2E-13 aprovado para iniciar/concluir análise TRAIN. A sessão confirma pipeline V4 em GPU AMD; a telemetria de percentual da GPU não está disponível. No primeiro teste E2E-02, os atalhos [ e ] para velocidade não responderam; o handler ignorava atalhos com SELECT/BUTTON/SUMMARY focado. Após ajustar o handler, o usuário confirmou o funcionamento dos dois atalhos e E2E-02 foi aprovado. No E2E-03, o usuário confirmou que rótulos GOAL/NO_GOAL/NOT_EVALUABLE, adiamento e filtros funcionaram corretamente. Nos E2E-04/E2E-05, o usuário confirmou persistência das notas/decisões e reabertura da sessão. A confirmação inline foi observada; a pedido do usuário, substituída por modal personalizado da aplicação, com nota bloqueada após salvar e botão Editar nota para reabrir. O usuário confirmou o modal personalizado e o fluxo Editar nota; E2E-04 aprovado. No E2E-06, a alteração de decisão foi confirmada e desfazer não está implementado. E2E-07 foi registrado como não implementado: não há cancelamento de análise, somente de compilação/download. Exportações HTML, CSV, JSON e MP4 aprovadas pelo usuário no E2E-10 em 22/10/2026.

**Atualização E2E-08 (20/10/2026):** o fixture MP4 inválido foi corretamente registrado como falha, sem sucesso falso. Como a primeira apresentação mostrava apenas uma mensagem genérica na tela, foi adicionado um modal personalizado com explicação simples e detalhes técnicos expansíveis; o usuário confirmou o funcionamento no reteste de 22/10/2026.

**Reteste E2E-08 (22/10/2026):** o usuário confirmou que o modal personalizado de erro funcionou; cenário aprovado.

**Reteste E2E-09 (22/10/2026):** o usuário confirmou que a sessão antiga concluída abriu e funcionou; compatibilidade aprovada.

**Reteste E2E-10 (22/10/2026):** o usuário confirmou que as exportações HTML, CSV, JSON e o MP4 compilado funcionaram corretamente.

**Reteste E2E-11 (22/10/2026):** o usuário confirmou que os modos apresentados são os esperados e que a sessão informa Pipeline V4.

**Reteste E2E-12 (22/10/2026):** conferência aprovada. A interface e `session.json` concordam em versão 0.4.0, pipeline V4, modo `alta_cobertura`, equipamento AMD Radeon RX 9060 XT e duração aproximada de 29:06. O trecho fornecido de `analysis/results.json` identifica Match_23/`alta_cobertura` e registra 60 candidatos selecionados agrupados em 48 segmentos. A telemetria de percentual de utilização da GPU está indisponível neste ambiente e foi documentada como limitação, sem invalidar os demais dados.

## 1. Objetivo editorial

Atualizar o manuscrito para refletir a avaliação final no split TEST, registrando a ressalva de exposição histórica de duas partidas, sem misturar populações nem apresentar a ferramenta como detector automático definitivo. O sistema deve ser descrito como um minerador que localiza e prioriza trechos `POSSIBLE_GOAL` para revisão humana.

Manter separados no artigo:

- Nested LOMO OOF em TRAIN: estimativa do procedimento de seleção por orçamento.
- LOMO usada para escolher as três configurações V3: resultado de seleção em TRAIN, sujeito a viés de seleção.
- Avaliação final em TEST: desempenho descritivo das três políticas fixas nos nove vídeos e 57 eventos catalogados cegamente.
- Implementação V4 da aplicação: produto operacional existente, distinto do executor experimental das políticas V3.

Não usar o resultado de TEST para ajustar hiperparâmetros, escolher uma política vencedora, retreinar modelos ou alterar a interface.

## 2. Mapa de alteração do artigo

O mapeamento foi conferido contra o fonte atual `docs/TCC_Matheus_Wastchuk.tex`; as alterações abaixo foram aplicadas ao manuscrito em 23/10/2026.

| Seção do artigo | Alteração indicada |
|---|---|
| Resumo / Abstract | Atualizar o objetivo para mineração assistida; incluir a população TEST e a cobertura dos três modos com ressalva de que cobertura não é precisão. |
| Introdução / objetivos | Reforçar o papel de apoio à revisão humana; retirar afirmações de reconhecimento automático ou economia de tempo humano não medida. |
| 4.1 Corpus | Informar a divisão TRAIN/VALIDATION/TEST conforme os protocolos, além dos 57 eventos anotados em TEST por uma pessoa usando os vídeos originais sem saídas da IA. |
| 4.2–4.4 Metodologia | Descrever freeze V3, anotação cega, instante visual do gol, regra de cobertura inclusiva e união de intervalos limitada à duração real do vídeo. |
| 5.4 Mineração e seleção | Atualizar modos V2 para FAST_V3/BALANCED_V3/HIGH_RECALL_V3; registrar RRF4_NET, janelas, K, gap e ausência de rescue. Dizer que esses modos foram avaliados pelo script experimental e não incorporados automaticamente ao app V4. |
| 5.5 Aplicação | Separar a aplicação V4 dos modos experimentais V3. A validação funcional manual E2E foi concluída em 13 cenários TRAIN; não é estudo controlado de usabilidade. |
| 6.6 Nested | Preservar os resultados OOF `105/125/132` em TRAIN25 para os tetos 30/40/58, identificando-os como estimativa Nested do procedimento. Manter `111/126/133` como métricas LOMO usadas na seleção V3, sujeitas a viés de seleção. |
| 6.8 TEST | Substituir o texto de “TEST pendente” pela tabela final abaixo, população, convenção e comparação pareada. |
| 6.9 Discussão | Discutir cobertura e minutos únicos como trade-off descritivo; abordar ganhos e perdas não aninhados entre modos, sem declarar vencedor. |
| 6.10 Limitações | Incluir exposição histórica de duas partidas TEST, anotação por operador único, ausência de avaliação de precisão e de tempo humano, além das limitações de domínio já documentadas. |
| Conclusão | Responder aos objetivos como sistema de priorização para revisão humana e distinguir os resultados Nested TRAIN, seleção TRAIN e cobertura TEST. |

## 3. Tabela final para a seção de resultados

| Política fixa | Cobertos / catálogo | Cobertura TEST | Minutos únicos médios por partida | Fração agregada dos vídeos originais | Teto por partida | Jogos acima do teto |
|---|---:|---:|---:|---:|---:|---:|
| FAST_V3 | 45/57 | 78,95% | 27,27 min | 34,68% | 30 min | 0/9 |
| BALANCED_V3 | 50/57 | 87,72% | 36,34 min | 46,21% | 40 min | 0/9 |
| HIGH_RECALL_V3 | 54/57 | 94,74% | 49,60 min | 63,07% | 58 min | 0/9 |

Os denominadores são eventos catalogados, não previsões positivas. A cobertura foi definida como pertença do timestamp visual humano a pelo menos um intervalo selecionado, com inclusão fechada (`início ≤ timestamp ≤ fim`) e tolerância adicional zero. Os minutos são a união dos trechos entregues, limitados à duração real dos vídeos.

Diferenças pareadas confirmadas em paired_mode_gains_test_v1.csv: BALANCED_V3 cobriu cinco gols não cobertos por FAST_V3, sem eventos exclusivos de FAST_V3 nessa comparação; HIGH_RECALL_V3 cobriu dez gols não cobertos por FAST_V3 e deixou de cobrir um coberto por FAST_V3; em relação a BALANCED_V3, HIGH_RECALL_V3 cobriu cinco eventos adicionais e deixou de cobrir um evento coberto por BALANCED_V3. Portanto, os conjuntos cobertos não são perfeitamente aninhados.

## 4. Texto sugerido para o resumo

> Foram avaliadas três políticas fixas de priorização de trechos em nove partidas do split TEST. Um operador anotou cegamente 57 instantes de gol a partir dos vídeos originais, sem consultar saídas do sistema. A cobertura dos timestamps foi de 78,95% para FAST_V3, 87,72% para BALANCED_V3 e 94,74% para HIGH_RECALL_V3. As médias de vídeo único disponibilizado por partida foram, respectivamente, 27,27, 36,34 e 49,60 minutos, dentro dos tetos operacionais de 30, 40 e 58 minutos em todos os nove jogos. Os resultados representam cobertura de eventos catalogados e não precisão de detecção automática; o sistema é proposto como apoio à revisão humana. Duas partidas tiveram exposição histórica anterior no projeto, limitação descrita no artigo.

Traduzir o resumo para o Abstract somente depois de aprovar a redação em português e conferir consistência com o restante do artigo.

## 5. Texto sugerido para a subseção de TEST

> A avaliação final foi conduzida uma única vez em nove partidas do split TEST, mantendo fixos os modelos, parâmetros de mineração e políticas de seleção V3. O conjunto de referência continha 57 instantes de gol anotados por um operador a partir dos vídeos originais, sem acesso a rankings, candidatos ou predições do sistema. Considerou-se um evento coberto quando seu timestamp pertencia a pelo menos um intervalo selecionado, incluindo as duas extremidades do intervalo. Intervalos sobrepostos ou contíguos foram unidos para calcular os minutos únicos de vídeo, com truncamento nos limites reais de cada arquivo.
>
> FAST_V3 cobriu 45 dos 57 eventos (78,95%), com média de 27,27 minutos únicos por partida; BALANCED_V3 cobriu 50 (87,72%), com 36,34 minutos; e HIGH_RECALL_V3 cobriu 54 (94,74%), com 49,60 minutos. Nenhum jogo excedeu o teto de 30, 40 ou 58 minutos correspondente. As diferenças pareadas mostram que a cobertura aumentou em termos agregados à medida que se passou a modos mais longos, embora os conjuntos de eventos cobertos não fossem perfeitamente aninhados. Esses resultados descrevem o comportamento das políticas fixas no split TEST e não foram usados para selecionar ou modificar uma configuração. Duas partidas tiveram exposição histórica anterior, conforme a limitação explicitada no artigo.

Antes da submissão, inserir uma tabela suplementar por partida, disponível em `metrics_v1/per_match_test_v1.csv`, e a tabela de eventos cobertos/perdidos em `metrics_v1/goal_coverage_test_v1.csv`.

## 6. Texto sugerido para discussão e limitações

> O modo HIGH_RECALL_V3 apresentou a maior cobertura agregada, mas também disponibilizou mais vídeo único para revisão. A cobertura não foi monotônica para cada evento: um gol coberto por FAST_V3 e um por BALANCED_V3 não foram cobertos por HIGH_RECALL_V3. Assim, a avaliação não sustenta a escolha de um modo universalmente superior; ela explicita alternativas operacionais com diferentes cargas de revisão.
>
> A métrica utilizada é cobertura de timestamps de gols catalogados. Ela não fornece precisão, taxa de falsos positivos confirmados por especialistas nem medida de economia de tempo humano. O catálogo foi produzido por um único operador, sem estimativa de concordância entre anotadores. Além disso, duas partidas TEST tiveram exposição histórica em etapas anteriores do projeto, o que deve ser tratado como ameaça à validade. A interface disponível corresponde ao fluxo V4; os três modos V3 foram avaliados em um executor experimental e não devem ser atribuídos à interface sem implementação e validação próprias.

Se o estudo com revisores não ocorrer, declarar explicitamente que o tempo humano de revisão não foi avaliado e tratar esse estudo como trabalho futuro.

## 7. Texto sugerido para conclusão

> Os resultados sustentam a viabilidade de minerar e priorizar trechos de vídeo para apoiar a revisão humana de eventos de gol. No conjunto TEST, as três políticas fixas cobriram de 78,95% a 94,74% dos timestamps catalogados dentro dos respectivos tetos de vídeo, mas com diferentes volumes de material para inspeção e casos não cobertos por cada modo. Como a avaliação mediu cobertura e não precisão ou tempo de trabalho humano, o sistema não substitui a decisão do revisor e não permite afirmar economia temporal. A validação de uso ponta a ponta da aplicação e um estudo controlado com revisores permanecem como etapas complementares.

## 8. Evidências que devem ser citadas ou preservadas

- Protocolo: [`test_evaluation_protocol_v1.md`](../../experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_evaluation_protocol_v1.md).
- Freeze e nota da revisão de `match_40`: [`test_catalog_freeze_manifest_v1.json`](../../experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_catalog_freeze_manifest_v1.json).
- Resultados integrais, hashes e tempos: [`test_v3_results_report.md`](../../experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/test_v3_results_report.md).
- Inferência por partida e candidatos: `attempt_02/inference_manifest_test_v1.json`, `ranked_candidates_test_v1.csv`, `selected_candidates_test_v1.csv` e `per_match_inference_test_v1.csv`.
- Métricas finais: `metrics_v1/summary_test_v1.json`, `goal_coverage_test_v1.csv`, `per_match_test_v1.csv` e `paired_mode_gains_test_v1.csv`.

## 9. Próximas etapas

1. [x] Localizar o arquivo-fonte atual e conferir suas seções.
2. [x] Incorporar os resultados ao resumo, Abstract, metodologia, resultados, discussão, limitações e conclusão em [`TCC_Matheus_Wastchuk.tex`](../TCC_Matheus_Wastchuk.tex).
3. Revisar a compilação LaTeX, a paginação e o conteúdo com o orientador antes da versão final.
4. **Concluído em 22/10/2026:** matriz E2E executada com vídeos TRAIN; resultados e evidências estão em [`application_e2e_matrix_2026-10-20.csv`](application_e2e_matrix_2026-10-20.csv). Os 13 cenários foram avaliados; E2E-06 registra que desfazer não está implementado e E2E-07 que não há cancelamento de análise. A telemetria de percentual de GPU permanece indisponível neste ambiente.
5. Se houver viabilidade e aprovação ética/institucional, planejar estudo-piloto de tempo humano separado, sem utilizar TEST para desenvolvimento de UX.
6. Após revisão com o orientador, preparar a versão final para orientação/defesa.

**Restrições metodológicas:** não reabrir TEST, não retreinar, não retunar e não escolher vencedor a partir destes resultados. Mudanças na aplicação que alterem ranking, seleção ou segmentos exigem novo protocolo e não podem ser apresentadas como parte desta avaliação concluída.
