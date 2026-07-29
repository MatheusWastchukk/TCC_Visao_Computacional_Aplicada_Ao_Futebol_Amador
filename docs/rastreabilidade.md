# Matriz de Rastreabilidade

Data de criacao: 29/07/2026

## Objetivos especificos

| ID | Objetivo especifico | Tarefas relacionadas | Evidencia esperada |
|---|---|---|---|
| OE-001 | Construir uma base organizada de videos de futebol amador para experimentos. | TCC-009, TCC-010, TCC-011, TCC-012, TCC-013 | `data/catalogo_videos.csv`, `data/metadata.csv`, `data/annotations/events.csv` |
| OE-002 | Implementar um pipeline de leitura, validacao e processamento frame a frame. | TCC-014, TCC-035 | Modulos em `src/`, CLI em `app/cli`, relatorio de compatibilidade |
| OE-003 | Detectar jogadores, bola e evidencias visuais nos clipes selecionados. | TCC-015, TCC-016, TCC-017, TCC-018 | Resultados estruturados e frames anotados em `outputs/frames` |
| OE-004 | Rastrear a trajetoria da bola e relaciona-la com a regiao do gol. | TCC-019, TCC-020, TCC-021, TCC-022 | Arquivos de trajetoria, configuracoes de gol e relatorios |
| OE-005 | Implementar e avaliar a abordagem heuristica de evento de gol. | TCC-023, TCC-024, TCC-025, TCC-031 | Saidas da Abordagem A e metricas finais |
| OE-006 | Implementar uma segunda abordagem comparavel, temporal ou hibrida. | TCC-026, TCC-027, TCC-028, TCC-029, TCC-032 | Saidas da Abordagem B e evidencias visuais |
| OE-007 | Comparar cientificamente as abordagens e documentar limitacoes. | TCC-008, TCC-030, TCC-033, TCC-034, TCC-045 | `docs/protocolo-avaliacao.md`, `outputs/reports/comparacao_final.md` |
| OE-008 | Entregar uma demonstracao reproduzivel para a banca. | TCC-035, TCC-036, TCC-037, TCC-038, TCC-040, TCC-041, TCC-048, TCC-049, TCC-051 | Interface, guia de instalacao, video, slides e pacote final |

## Requisitos obrigatorios

| Requisito | Tarefa principal | Teste ou comprovacao | Evidencia |
|---|---|---|---|
| RF-001 | TCC-014, TCC-035 | Abrir video local valido pelo pipeline ou CLI. | Log de execucao |
| RF-002 | TCC-014 | Executar com arquivo valido, inexistente e invalido. | Testes de validacao |
| RF-003 | TCC-014 | Comparar metadados com OpenCV ou `ffprobe`. | Relatorio de compatibilidade |
| RF-004 | TCC-014 | Conferir contagem de frames processados. | Log ou CSV de processamento |
| RF-005 | TCC-015 | Revisar amostra de frames com jogadores detectados. | `outputs/frames` |
| RF-006 | TCC-015, TCC-016, TCC-018 | Medir taxa de frames com bola detectada. | Relatorio baseline |
| RF-007 | TCC-019 | Exportar trajetoria com frame, timestamp e centro da bola. | CSV/JSON de tracking |
| RF-008 | TCC-020, TCC-021 | Carregar configuracao de poligono do gol. | Arquivos em `configs/` |
| RF-009 | TCC-023, TCC-024, TCC-025 | Rodar regras da Abordagem A em clipes anotados. | Relatorio da Abordagem A |
| RF-010 | TCC-026, TCC-027, TCC-028, TCC-029 | Rodar Abordagem B no mesmo formato de saida. | Relatorio da Abordagem B |
| RF-011 | TCC-025, TCC-029 | Conferir frame representativo para cada evento candidato. | `outputs/frames/approach_a`, `outputs/frames/approach_b` |
| RF-012 | TCC-008, TCC-031, TCC-032, TCC-033 | Conferir metricas exportadas e calculos. | `outputs/reports` |
| RF-013 | TCC-036, TCC-037, TCC-038 | Completar fluxo de demonstracao local. | Screenshots e video curto |
| RNF-001 | TCC-004, TCC-040 | Executar instalacao e testes no Windows 11. | `docs/environment-check.md` |
| RNF-002 | TCC-004, TCC-005 | Executar verificacao e inferencia em CPU. | `experiments/hardware-spike/relatorio.md` |
| RNF-003 | TCC-014, TCC-035, TCC-037 | Simular erros de entrada e conferir mensagens. | Testes e logs |
| RNF-004 | TCC-003, TCC-004, TCC-030, TCC-040 | Versionar dependencias, configuracoes e manifests. | Git, `requirements.txt`, documentos |
| RNF-005 | TCC-020, TCC-021, TCC-024, TCC-041 | Validar uso de arquivos em `configs/`. | Configuracoes versionadas |
| RNF-006 | TCC-011, TCC-014, TCC-038 | Executar somente com arquivos locais. | Checklist de demonstracao |
| RNF-007 | TCC-011, TCC-014 | Validar duracao dos clipes selecionados. | `data/metadata.csv` |
| RNF-008 | TCC-001, TCC-030, TCC-051 | Conferir `git status` e `.gitignore` antes de commits. | Historico Git |

## Metricas

| Metrica | Experimento ou tarefa | Metodo de coleta | Evidencia |
|---|---|---|---|
| Precisao de evento | TCC-031, TCC-032, TCC-033 | Matriz de confusao contra `data/annotations/events.csv`. | Relatorio final |
| Revocacao de evento | TCC-031, TCC-032, TCC-033 | Matriz de confusao contra `data/annotations/events.csv`. | Relatorio final |
| F1-score de evento | TCC-031, TCC-032, TCC-033 | Media harmonica entre precisao e revocacao. | Relatorio final |
| Taxa de frames com bola detectada | TCC-015, TCC-018 | Frames com deteccao de bola / frames processados. | Relatorio baseline |
| Continuidade da trajetoria | TCC-019, TCC-031 | Percentual de frames sucessivos com posicao estimada. | CSV de tracking |
| Interrupcoes de tracking | TCC-019, TCC-031 | Contagem de lacunas maiores que limite configurado. | CSV de tracking |
| Tempo total | TCC-005, TCC-031, TCC-032 | Cronometragem da execucao por clipe e hardware. | Relatorios de experimento |
| FPS de processamento | TCC-005, TCC-031, TCC-032 | Frames processados / tempo total. | Relatorios de experimento |
| Uso aproximado de memoria | TCC-005, TCC-031, TCC-032 | Registro do sistema ou biblioteca de monitoramento. | Relatorio de hardware |

## Checklist de aceite da TCC-003

- Nenhum objetivo especifico ficou sem tarefa correspondente.
- Nenhum requisito obrigatorio ficou sem teste ou comprovacao.
- Nenhuma metrica ficou sem metodo de coleta.
- A matriz deve ser atualizada quando novos requisitos, tarefas ou metricas forem alterados.
