# TCC — Detecção e Análise de Vídeos de Futebol

Projeto de conclusão de curso sobre mineração de trechos potencialmente relacionados a gols em vídeos completos de futebol, com priorização para revisão humana por meio de técnicas de visão computacional.

O título oficial do trabalho é **Detecção e Análise de Eventos em Vídeos de Futebol Utilizando Técnicas de Visão Computacional**. O título é mantido conforme a definição institucional. No texto, o escopo metodológico é delimitado como mineração e priorização de candidatos, sem alegação de confirmação automática de gols.

## Pergunta de pesquisa

Em partidas completas do corpus, quanto da cobertura dos timestamps de gols catalogados pode ser preservada por políticas fixas, selecionadas em TRAIN e aplicadas em TEST, sob diferentes tetos de minutos únicos de vídeo por partida?

## Contribuição e estado

O projeto reúne:

- um pipeline local para gerar e ranquear candidatos temporais;
- avaliação Nested LOMO em 25 partidas TRAIN do procedimento de seleção sob orçamento;
- avaliação final congelada das políticas experimentais V3 em nove partidas TEST e 57 timestamps;
- uma aplicação Electron/React + FastAPI para inspecionar, anotar e exportar candidatos pelo fluxo operacional V4.

Resultados V3 em TEST: FAST_V3 cobriu 45/57 timestamps (78,95%) com média de 27,27 minutos únicos por partida; BALANCED_V3, 50/57 (87,72%) com 36,34 minutos; HIGH_RECALL_V3, 54/57 (94,74%) com 49,60 minutos. Nenhum jogo excedeu o teto de vídeo do respectivo modo.

Esses resultados medem cobertura temporal de timestamps catalogados. Não medem precisão de detecção de gols, taxa de falsos positivos ou tempo humano de revisão. As políticas experimentais V3 são distintas dos modos operacionais V4 da aplicação.

## Estrutura

- `app/`: aplicação e ferramentas de análise/revisão.
- `backend/`, `desktop/`: API local e aplicação Electron.
- `configs/`: reservado para configurações; os manifests de experimento ficam junto aos experimentos.
- `data/`: vídeos e anotações locais, não incluídos no pacote.
- `docs/`: protocolos, resultados, dependências, instruções do ambiente e rastreabilidade.
- `experiments/`: scripts, manifests e resultados experimentais.
- `models/`, `weights/`: pesos locais não incluídos no pacote.
- `src/`: módulos de eventos usados pela aplicação.
- `tests/`: testes automatizados existentes.

## Evidências principais

- Resultados finais V3 TEST: `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/`.
- Nested LOMO TRAIN: `docs/plans/95_resultados_nested_lomo_fullpool_25_train_14-10-2026.md`.
- Modos V3: `docs/plans/99_resultados_selecao_modos_lomo_no_rescue_v3_14-10-2026.md`.
- Validação manual E2E da aplicação: `docs/plans/application_e2e_matrix_2026-10-20.csv`.
- Plano e manifesto preliminar de reprodução: `docs/entrega_reprodutivel.md`, `docs/entrega_reprodutivel_manifesto_v1.json` e `docs/ambiente_reproducao.md`.
- Auditoria preliminar de licenças e direitos: `docs/auditoria_licencas_pacote.md`.
- Plano condicional do piloto humano: `docs/plans/104_plano_piloto_revisao_humana_train.md`.
- Requisitos e rastreabilidade: `docs/requisitos.md` e `docs/rastreabilidade.md`.

## Execução e dependências

O aplicativo Electron atual depende de Node.js/npm, Python, PyTorch, modelos locais e FFmpeg. O ambiente de desenvolvimento foi configurado para Windows e GPU AMD/ROCm. O build Vite compila a interface; ele não empacota o ambiente Python, os pesos nem todo o runtime de FFmpeg.

Há em `release/GoalEventReview/` uma saída antiga PyInstaller baseada em PySide6. Ela não corresponde ao aplicativo Electron V4 validado em outubro e não deve ser usada como prova de pacote portátil atual.

Consulte `docs/entrega_reprodutivel.md` para preparar uma cópia do repositório. A escrita e compilação do artigo acontecem no Overleaf e não são etapas deste ambiente local.

## Documentação de origem

Requisitos e rastreabilidade antigos foram revisados para corresponder ao escopo final do TCC. Relatórios numerados de experimentos permanecem como histórico técnico; sua presença não significa que cada hipótese ou configuração tenha sido promovida ao sistema final.
