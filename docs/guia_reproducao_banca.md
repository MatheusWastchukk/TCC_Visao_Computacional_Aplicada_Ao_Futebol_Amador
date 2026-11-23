# Guia de versão e reprodução para a banca

**Revisão documental:** 23/11/2026.

## Identificação da versão

| Item | Identificação |
|---|---|
| Base de código conferida | `265b95e5e35a6ab4816a00de7d0b2023c9a6303b` |
| Estado na abertura desta revisão | Árvore Git limpa; os ajustes documentais de 23/11 são posteriores à base acima e ainda não estavam commitados. |
| Aplicação | `0.4.3`, conforme `desktop/package.json`; fluxo operacional V4. |
| Avaliação final do artigo | Políticas congeladas `FAST_V3`, `BALANCED_V3` e `HIGH_RECALL_V3`; nove partidas TEST e 57 instantes catalogados. |
| Inventário de outubro | `entrega_reprodutivel_manifesto_v1.json`: histórico `DRAFT_NOT_A_RELEASE`, não manifesto desta revisão. |

A versão da aplicação, a família de políticas e o commit identificam aspectos diferentes. Os resultados TEST V3 não constituem avaliação dos modos operacionais V4.

Antes de fechar a cópia de entrega, registre nela a saída dos comandos abaixo. Se houver mudanças locais, o SHA de HEAD identifica apenas a base; inclua também os arquivos alterados e seus hashes no manifesto da cópia. Este guia não afirma que um pacote final já foi gerado ou distribuído.

```powershell
git rev-parse HEAD
git status --short
```

## Rota mínima: tabelas finais TEST

Esta rota usa Python 3.12 e apenas sua biblioteca padrão. Não precisa instalar PyTorch, Ultralytics, a aplicação ou FFmpeg. Não executa inferência, não carrega vídeos e não seleciona novas políticas.

Execute da raiz do repositório. Preserve a estrutura de diretórios, pois o agregador resolve caminhos relativos à raiz e o manifesto do catálogo aponta para o CSV congelado. O diretório de saída deve estar ausente ou vazio.

### Sete arquivos necessários

| Papel | Caminho relativo à raiz |
|---|---|
| Agregador | `experiments/ball/evaluate_frozen_v3_test_metrics_v1.py` |
| Manifesto do catálogo | `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_catalog_freeze_manifest_v1.json` |
| Catálogo humano congelado | `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/goal_catalog_timestamp_manifest_frozen_v1.csv` |
| Manifesto da inferência válida | `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/inference_manifest_test_v1.json` |
| Candidatos ranqueados | `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/ranked_candidates_test_v1.csv` |
| Candidatos selecionados | `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/selected_candidates_test_v1.csv` |
| Informações por partida | `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/per_match_inference_test_v1.csv` |

Os hashes registrados na reprodução de outubro estão no [Relatório 106](plans/106_reproducao_limpa_metricas_test_v3_26-10-2026.md#entradas-e-integridade). O agregador verifica o SHA-256 do catálogo contra seu manifesto. Confirme a presença das sete entradas na cópia entregue; os arquivos derivados só devem acompanhá-la conforme as autorizações documentadas no [plano de reprodução](entrega_reprodutivel.md).

### Comando

Com Python 3.12 instalado e disponível como `python`:

```powershell
python --version
python .\experiments\ball\evaluate_frozen_v3_test_metrics_v1.py `
  --inference .\experiments\ball\TEST-V3-FINAL-EVALUATION-20261019\attempt_02 `
  --output .\.tmp\metricas_banca_20261123 `
  --catalog-freeze .\experiments\ball\GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3\test_catalog_freeze_manifest_v1.json
```

Neste checkout, também é possível substituir `python` por `& .\.venv-rocm\Scripts\python.exe`; esse ambiente local não precisa integrar a entrega. Para repetir o cálculo, escolha outro diretório de saída vazio, preservando as referências em `metrics_v1/`.

### Mapa entre resultados, arquivos e artigo

As quatro saídas são gravadas no diretório passado em `--output`. As referências ficam em `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/metrics_v1/`.

| Resultado no artigo | Saída do agregador | Localização estável no LaTeX |
|---|---|---|
| Cobertura agregada, duração média e respeito aos tetos | `summary_test_v1.json` | Tabela “Cobertura das políticas fixas V3 no conjunto TEST”, rótulo `tab:test-final`. |
| Cobertura e minutos por partida | `per_match_test_v1.csv` | Tabela “Cobertura por partida e minutos únicos entregues”, rótulo `tab:test-per-match`. |
| Eventos exclusivos e compartilhados entre políticas | `paired_mode_gains_test_v1.csv` | Discussão após a tabela por partida, na subseção de avaliação final TEST. |
| Rastreabilidade de cada instante coberto | `goal_coverage_test_v1.csv` | Evidência de suporte às contagens das duas tabelas TEST. |

Os rótulos e títulos permitem localizar as tabelas mesmo se a numeração mudar no PDF. Este mapa não cobre as tabelas Nested TRAIN ou as auditorias de detectores; elas têm protocolos próprios, indicados no README e nos relatórios correspondentes.

### Valores esperados e aceite

| Política | Cobertura | Minutos únicos médios por partida | Teto por partida | Partidas acima do teto |
|---|---|---|---|---|
| FAST_V3 | 45/57 (78,95%) | 27,27 | 30 min | 0/9 |
| BALANCED_V3 | 50/57 (87,72%) | 36,34 | 40 min | 0/9 |
| HIGH_RECALL_V3 | 54/57 (94,74%) | 49,60 | 58 min | 0/9 |

O resumo deve declarar `TEST_V3_METRICS_COMPLETE`, nove partidas e 57 eventos. Compare contagens, categorias e valores publicados com as referências. Para floats não arredondados, use tolerância absoluta de `1e-12`; caminhos absolutos dos manifests variam conforme a máquina e não são métricas. A reprodução histórica e as diferenças de arredondamento estão descritas no Relatório 106.

**Verificação local em 23/11/2026:** o comando foi executado com Python 3.12.10 do ambiente local, gravando em `.tmp/metricas_banca_20261123/`. O resumo e os três CSVs coincidiram semanticamente com as referências: 57 linhas por evento, nove por partida e três comparações pareadas. A maior diferença numérica observada foi `5.684341886080802e-14`, abaixo da tolerância. As sete entradas existem, estão versionadas e seus SHA-256 coincidiram com os registros do Relatório 106. Essa conferência não executou inferência nem validou uma nova instalação do aplicativo.

Cobertura significa inclusão do instante catalogado em um intervalo selecionado, com extremidades incluídas e tolerância adicional zero. Minutos únicos contabilizam uma única vez as sobreposições. Esses resultados não medem precisão de classificação de gols ou tempo humano economizado.

## Outras formas de execução

- **Aplicação e instalação:** [build CPU](build_cpu.md) e [instalador online 0.4.3](instalador_online_043.md), com versões, hashes e limites de validação por revisão. A mineração V4 usa Ultralytics, incluído em `requirements-cpu.txt`.
- **Ambiente de desenvolvimento:** [ambiente de reprodução](ambiente_reproducao.md). Uma receita documentada não equivale a uma instalação limpa validada em qualquer computador.
- **Reexecução completa da avaliação:** exige vídeos autorizados, pesos exatos, configurações congeladas e ambiente compatível. O recálculo acima não demonstra reprodução da inferência completa e não autoriza retuning de TEST.
- **Distribuição:** [auditoria preliminar de licenças e direitos](auditoria_licencas_pacote.md). O guia é um mapa técnico e não encerra as pendências de autorização do pacote.
