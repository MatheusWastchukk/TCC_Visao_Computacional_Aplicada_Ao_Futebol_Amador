# Relatório 101 — Auditoria pré-TEST do freeze V3

**Data:** 17/10/2026<br>
**Base:** Relatório 100 e freeze `REPORT87-PD-SPRINT-FINAL-FREEZE-20261011` V3.<br>
**Estado:** auditoria documental e de hashes; nenhum vídeo, timestamp, score ou saída de modelo de TEST foi consultado.

## Resultado

### Integridade confirmada

- `final_freeze_manifest_v3.json` existe, tem `manifest_version=3`, declara supersessão de `final_freeze_manifest_v2.json` e preserva o Nested 25/25 como evidência principal.
- O preflight `finalize_report98_modes_v3.py --preflight` passou em 17/10/2026: 72 configurações da grade, 1.800 linhas OOF, 25 partidas, 152 gols e 4 hashes de modelos verificados; `test_accessed=false`.
- Os modos registrados são distintos: `FAST_V3` = RRF4_NET, janela 30 s, K=60, gap 45 s; `BALANCED_V3` = RRF4_NET, janela 40 s, K=60, gap 45 s; `HIGH_RECALL_V3` = RRF4_NET, janela 60 s, K=55, gap 45 s.
- Os quatro modelos operacionais são reutilizados sem alteração do pacote `REPORT94-FULLPOOL-FINAL-MODES-V2`, com os hashes verificados no manifesto V3.
- O código e as tabelas experimentais do V3 constam por hash no snapshot de origem `interim_freeze_manifest_v49.json`.

### Bloqueio identificado para a inferência V3

O projeto não contém um executor de avaliação TEST que carregue diretamente `final_modes_v3.json`. A busca no código de aplicação não encontrou referência a esse arquivo. `app/analyze_match.py` importa `src.events.operational_modes_v4.MODES`, oferece três modos V4 diferentes e, no fluxo de vídeo, carrega os pesos definidos pelo pipeline V4. Seu caminho padrão restringe a inferência a TRAIN; a opção de vídeo externo ainda executa V4. Portanto, usar o app atual não equivale a executar FAST/BALANCED/HIGH_RECALL V3.

Também não há, até esta auditoria, um scorer experimental end-to-end que gere em TEST o mesmo pool de candidatos, os escores temporal/visual/fusão/net-motion e os segmentos usados pela seleção Nested. Os 30/40/58 minutos são tetos de elegibilidade observados na seleção TRAIN; não há seletor operacional com garantia de teto para partidas novas. O resultado de TEST deve medir e relatar qualquer excedente sem alterar K, janelas ou ranking.

**Decisão do gate P0-A:** hashes e parâmetros passaram; capacidade de execução V3 não passou. Não executar inferência TEST até criar e auditar o scorer V3 isolado. Esta pendência não altera os modos congelados e não impede a preparação/anotação humana cega do catálogo.

## Rastreamento da versão

- `HEAD` observado: `92a080dab4fc136edc6430f5e15366d1b0f4d1b7` (`feat(app): ajusta aplicação para refletir melhor os rankings e melhorar a usabilidade`).
- A árvore de trabalho contém arquivos experimentais e documentais locais não commitados. Para a linhagem V3, usar os hashes SHA-256 do freeze V3/snapshot V49; não declarar que os scripts V3 pertencem ao commit HEAD.
- O pacote V4 permanece inalterado. Esta auditoria não renomeia nem promove V3 para a interface.

## Próximos passos

1. Registrar e executar o protocolo cego do Relatório 102, usando somente os nove vídeos originais e os 57 IDs catalogados.
2. Antes da inferência, construir o scorer experimental V3, demonstrar que carrega a configuração congelada e comparar seu mecanismo de seleção/união com as funções já usadas no Nested, sem ler rótulos de TEST.
3. Reexecutar um preflight de integridade e gerar o protocolo de avaliação TEST com hashes antes do primeiro score de TEST.
4. Só após o catálogo humano auditado e o executor aprovado, realizar a avaliação única dos três modos V3.
