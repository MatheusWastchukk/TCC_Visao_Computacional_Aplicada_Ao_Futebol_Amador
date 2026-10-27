# Rastreabilidade da pergunta, objetivos e evidências

**Título oficial (mantido):** Detecção e Análise de Eventos em Vídeos de Futebol Utilizando Técnicas de Visão Computacional.

## Pergunta de pesquisa

Em partidas completas do corpus, quanto da cobertura dos timestamps de gols catalogados pode ser preservada por políticas fixas, selecionadas em TRAIN e aplicadas em TEST, sob diferentes tetos de minutos únicos de vídeo por partida?

## Objetivo geral

Desenvolver e avaliar um protótipo que minera e prioriza trechos potencialmente relacionados a gols em gravações completas de futebol, apoiando a inspeção humana e mensurando a cobertura temporal de eventos catalogados em relação ao volume de vídeo selecionado.

## Matriz de rastreabilidade

| Objetivo | Evidência principal | Conjunto/unidade | Situação e limite |
|---|---|---|---|
| OE-01. Organizar um corpus de partidas completas com catálogos e divisão por partida. | Manifestos do corpus e tabela de corpus em `docs/TCC_Matheus_Wastchuk.tex`. | 42 partidas; 25 TRAIN, 8 VALIDATION, 9 TEST; 264 eventos catalogados. | Organizado; vídeos locais não acompanham a entrega por padrão. |
| OE-02. Investigar detecção da estrutura do gol e da bola, incluindo falhas de identidade. | Relatórios de goal geometry, bola, tracking e auditorias no histórico. | Amostras e populações indicadas por cada experimento. | Investigado; detector/tracker não é tratado como confirmação automática de gol. |
| OE-03. Gerar, agrupar e ranquear janelas candidatas. | Scripts e resultados em `experiments/ball/`; implementação local em `app/` e `src/events/`. | Grupos temporais e candidatos por partida. | Implementado para os fluxos registrados; versões experimentais e V4 operacional são distintas. |
| OE-04. Estimar em TRAIN o compromisso entre seleção, cobertura e custo de vídeo. | Relatórios 95 e 99; Nested LOMO e seleção V3. | 25 partidas, 152 eventos avaliáveis; cobertura e minutos únicos por orçamento. | Executado; resultados LOMO de seleção não são estimativas independentes. |
| OE-05. Avaliar políticas fixas em partidas TEST reservadas. | Protocolo congelado, manifests e `TEST-V3-FINAL-EVALUATION-20261019/metrics_v1/`. | 9 partidas, 57 timestamps; eventos cobertos e minutos únicos. | Executado uma vez; resultado descritivo, com exposição histórica de duas partidas e anotador único. |
| OE-06. Disponibilizar ferramenta local para revisão humana e registro de decisões. | Aplicação Electron/React + FastAPI e matriz E2E em `docs/plans/application_e2e_matrix_2026-10-20.csv`. | 13 cenários E2E. | Undo persistente e cancelamento implementados; build/checagens focadas passaram; usuário confirmou os retestes manuais de E2E-06/07 em 25/10. |
| OE-07. Documentar limites e preservar interpretação correta. | Artigo, relatórios, protocolos e matriz de requisitos. | Evidência separada por população e protocolo. | Em atualização final; sem alegação de precisão ou economia de tempo humano. |
| OE-08. Preparar entrega reproduzível. | `docs/entrega_reprodutivel.md` e `docs/entrega_reprodutivel_manifesto_v1.json`. | Código, manifests, dependências, configurações, hashes e exclusões. | Plano e manifesto preliminar criados; snapshot final e pacote precisam ser materializados/verificados. |
| OE-09. Avaliar exploratoriamente o fluxo humano assistido, se aprovado e viável. | Plano de piloto TRAIN-only em `docs/plans/104_plano_piloto_revisao_humana_train.md`. | Revisores, tempo ativo, eventos recuperados, falsos apontamentos e interação. | Condicional; não executado, depende de análise institucional e prazo. |

## Interpretação da métrica principal

Um evento catalogado é coberto quando seu timestamp humano congelado pertence a pelo menos um intervalo selecionado, usando inclusão fechada: início do intervalo ≤ timestamp ≤ fim do intervalo. A cobertura divide o número de timestamps avaliáveis cobertos pelo total de timestamps avaliáveis do protocolo. A avaliação final TEST usa tolerância adicional zero.

Cobertura do Candidate Pool utiliza todas as janelas antes do corte. Cobertura da shortlist utiliza somente as janelas entregues pela política. Nenhuma das duas medidas fornece, por si, precisão de detecção ou taxa de falsos positivos. Minutos únicos são a duração da união dos intervalos e não equivalem ao tempo humano para revisar.

## Hierarquia dos resultados

1. **Nested LOMO TRAIN:** estimativa do procedimento de seleção em folds por partida.
2. **Seleção LOMO TRAIN:** valores usados para escolher configurações; sujeitos a viés de seleção.
3. **TEST V3:** cobertura descritiva de políticas fixas em timestamps congelados.
4. **V4 VALIDATION:** evidência operacional histórica em conjunto consultado durante o desenvolvimento.
5. **E2E da aplicação:** verificação funcional manual; não mede desempenho científico nem usabilidade geral.
6. **Piloto com revisores:** futuro, exploratório e TRAIN-only, se aprovado.
