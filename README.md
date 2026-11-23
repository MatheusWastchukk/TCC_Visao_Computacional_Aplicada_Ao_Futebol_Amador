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

## Versão e reprodução para a banca

O [guia de reprodução para a banca](docs/guia_reproducao_banca.md), revisado em 23/11/2026, identifica a base de código, a aplicação `0.4.3` e as políticas experimentais V3.

Para reproduzir as tabelas finais TEST, o guia lista as sete entradas necessárias, o comando Python e as quatro saídas esperadas. A rota recalcula métricas a partir de candidatos e catálogo congelados, sem vídeos, pesos ou nova inferência. O manifesto preliminar de outubro é histórico e não identifica o pacote atual. A inclusão dos dados derivados na entrega continua sujeita às autorizações registradas no plano de reprodução.

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

A preparação do pacote Windows/CPU está documentada em [docs/build_cpu.md](docs/build_cpu.md).
O backend de linha de comando usa CPU por padrão. O build CPU reúne backend, modelos e FFmpeg em
um pacote preliminar para teste; a validação em outro computador permanece separada
da compilação local.

O pacote completo 0.4.2 inclui os três ambientes e apresenta CPU, GPU NVIDIA e
GPU AMD na abertura, com validação antes de iniciar a análise. Construção,
preparação offline e limites de compatibilidade estão em
[docs/processing_modes_042.md](docs/processing_modes_042.md).

O instalador completo para testes, com atalhos e desinstalador, é construído
conforme [docs/instalador_windows_042.md](docs/instalador_windows_042.md).

O instalador 0.4.3 de 404,6 MB, com downloads GPU oficiais durante a instalação,
está descrito em [docs/instalador_online_043.md](docs/instalador_online_043.md),
incluindo a validação local e os limites de compatibilidade.

Os ajustes de uso da GPU AMD e do painel recolhível de métricas estão em
[docs/metricas_processamento.md](docs/metricas_processamento.md), com o estado de validação e empacotamento.
