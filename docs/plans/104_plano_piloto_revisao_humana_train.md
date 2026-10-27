# Plano condicional de piloto de revisão humana assistida — TRAIN

**Estado:** planejamento; estudo não iniciado.<br>
**População permitida:** somente partidas do conjunto TRAIN.<br>
**Prazo de entrega do TCC:** 23/11/2026.<br>
**Condição de início:** confirmação do orientador e verificação dos requisitos éticos/institucionais aplicáveis antes de convidar participantes ou coletar dados.

## 1. Pergunta exploratória

Em uma amostra pequena de partidas TRAIN, como diferem o tempo ativo de revisão e a recuperação de eventos catalogados quando uma pessoa revisa o vídeo completo em sequência ou inspeciona candidatos produzidos pela aplicação V4?

O piloto avalia o fluxo de revisão da aplicação V4 disponível. Ele não avalia as políticas experimentais V3, não demonstra precisão automática e não modifica o protocolo de TEST já encerrado.

## 2. Objetivo e limites

Observar se a priorização de candidatos ajuda pessoas a localizar eventos catalogados e caracterizar o custo de revisão, os eventos não encontrados e as interações necessárias. O estudo será piloto de viabilidade e produzirá resultados descritivos. Uma amostra de três a cinco pessoas não sustentará uma alegação estatística geral de economia ou melhora.

Não alterar detector, ranking, janelas, seleção ou parâmetros para preparar o estudo. Não consultar vídeos, timestamps, rankings ou métricas de TEST para selecionar jogos, desenhar tarefas ou melhorar a interface.

## 3. Desenho proposto

- **Participantes:** alvo inicial de 3–5 voluntários, com experiência declarada em futebol e/ou revisão de vídeo registrada de modo breve e não identificável.
- **Unidade de tarefa:** uma partida completa TRAIN atribuída a uma condição de revisão.
- **Condição manual:** revisar a gravação completa em ordem cronológica e registrar os momentos que a pessoa considera gols.
- **Condição assistida:** revisar candidatos da versão V4 da aplicação e registrar as decisões humanas disponíveis na interface.
- **Comparação intra participante:** cada pessoa executa uma tarefa manual e uma assistida, com partidas distintas de dificuldade aproximada; a ordem das condições é contrabalançada entre participantes.
- **Seleção de partidas:** escolher previamente um conjunto pequeno de partidas TRAIN com base apenas em metadados do corpus, duração e número de eventos catalogados. Distribuir as mesmas partidas entre condições de forma balanceada entre participantes. Não usar TEST.
- **Treinamento:** demonstração curta do procedimento antes de iniciar a cronometragem. Não revelar timestamps ou rótulos do catálogo aos participantes.
- **Pausa e processamento:** separar tempo de processamento automático, tempo ativo de revisão e pausas. Tempo de inferência não é tempo humano de revisão.

## 4. Medidas

### Medida primária descritiva

- Minutos ativos desde o início da tarefa até a pessoa concluir a revisão da partida ou declarar que não continuará.
- Relatar por participante e condição; apresentar mediana, intervalo e valores individuais, sem inferência populacional.

### Medidas secundárias

- Número de eventos do catálogo TRAIN encontrados pela pessoa.
- Número de marcações humanas de gol sem correspondência no catálogo, rotuladas como **marcações não correspondentes ao catálogo**, não como falsos positivos confirmados.
- Número de eventos catalogados não encontrados.
- Número de candidatos revisados, decisões, revisitas e pausas.
- Observações breves de dificuldade e uma avaliação pós tarefa de facilidade/confiança, claramente identificada como exploratória.

O catálogo TRAIN pode ser incompleto. Uma marcação sem correspondência no catálogo não será chamada de falso positivo sem uma verificação humana independente. Se algum evento adicional for descoberto, registrá-lo separadamente e não alterar retroativamente os denominadores congelados sem protocolo e decisão documentados.

## 5. Registro e proteção dos participantes

- Consultar orientador e instância institucional responsável sobre aprovação, consentimento e armazenamento antes do recrutamento.
- Preparar uma folha informativa e termo de consentimento conforme o procedimento institucional aplicável.
- Usar códigos anônimos de participante; não registrar nome ou dado pessoal desnecessário.
- Registrar data, condição, partida, ordem, horários de início/fim, pausas e problemas técnicos em formulário padronizado.
- Guardar respostas em área restrita e descrever no TCC apenas dados agregados compatíveis com o consentimento e as regras institucionais.

## 6. Plano de análise

- Descrever o procedimento, o número de participantes e tarefas completas/incompletas.
- Mostrar tempos individuais e diferença observada entre condições por participante.
- Reportar eventos encontrados/não encontrados em relação ao catálogo e as marcações não correspondentes em separado.
- Distinguir o tempo de processamento V4, vídeo disponibilizado e tempo humano ativo.
- Não usar teste de significância ou linguagem de eficácia geral para esta amostra pequena.
- Se houver perdas, falhas da aplicação ou mudanças no protocolo, registrar e não excluir silenciosamente observações.

## 7. Critérios de GO/NO-GO e cronograma

| Data-alvo | Gate/atividade | Saída |
|---|---|---|
| Até 31/10/2026 | Confirmar com orientador a utilidade acadêmica e verificar processo institucional, participantes e consentimento. | GO condicionado ou NO-GO documentado. |
| 01–07/11/2026 | Somente após GO: fechar protocolo, partidas TRAIN, formulários e roteiro; testar fluxo sem usar TEST. | Protocolo e materiais aprovados. |
| 08–14/11/2026 | Somente se autorização e agenda permitirem: coletar piloto com 3–5 participantes. | Registros anônimos completos e incidentes documentados. |
| 15–18/11/2026 | Consolidar resultados descritivos e decidir com orientador se cabem no texto ou se ficam como trabalho futuro. | Tabela exploratória ou justificativa de não execução. |
| A qualquer momento | Se autorização, recrutamento ou tempo ameaçar a revisão final do artigo, interromper o plano antes da coleta ou encerrá-lo. | Registrar NO-GO sem impacto no resultado algorítmico. |

**Gate obrigatório:** não iniciar recrutamento nem coleta sem confirmar o procedimento ético/institucional aplicável. Não atrasar compilação, revisão do artigo, orientação ou entrega para concluir o piloto.

## 8. Texto de contingência se não for executado

“Um estudo com participantes para medir o tempo ativo, a recuperação de eventos e a experiência de revisão não foi conduzido neste trabalho. Os minutos únicos apresentados quantificam o volume de vídeo disponibilizado e não equivalem ao tempo de trabalho humano. A avaliação controlada do fluxo assistido permanece como trabalho futuro.”\n
