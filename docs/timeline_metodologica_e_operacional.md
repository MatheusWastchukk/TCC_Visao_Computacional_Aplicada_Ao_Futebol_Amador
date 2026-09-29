# Timeline metodológica e operacional

Este artefato organiza a evolução do projeto para apresentação e artigo. Ele
separa mudanças de método, validações operacionais e desenvolvimento da
aplicação. Métricas de `TRAIN` não são usadas como avaliação geral; `TEST`
permanece fechado até a avaliação final autorizada.

| Período | Marco | Resultado verificável | Limite preservado |
|---|---|---|---|
| 2025--início de 2026 | Preparação de dados e detectores de estrutura/bola | Dataset, experimentos de detecção e registros reproduzíveis | Detector isolado não decide gol. |
| 2026-09 | Pipeline temporal V4 | Coocorrências bola--gol são agrupadas, ranqueadas e transformadas em candidatos | A saída é `POSSIBLE_GOAL`, não confirmação automática. |
| 2026-09 | Modos operacionais V4 | Três listas de revisão: alta cobertura, cobertura reduzida e triagem rápida | A configuração V4 foi congelada; comparações posteriores devem declarar se são experimentais. |
| 2026-09 | Smokes operacionais em TRAIN | Duas partidas TRAIN revisadas ponta a ponta e associadas ao catálogo | São demonstrações de fluxo, não estimativa generalizável. |
| 2026-09 | Auditoria de extremos de ranking | Casos de bola difícil, trajetória difícil e contexto confuso foram registrados | A auditoria orienta hipóteses, sem alterar o resultado congelado. |
| 2026-09 | Aplicação desktop | Electron/React e API Python local processam vídeo arbitrário, exibem clipes e persistem tags de revisão | A classificação humana continua separada da inferência. |
| Próximo ciclo | Trajetória local consistente | Experimento isolado para reduzir saltos da bola para objetos aleatórios perto de candidatos | Só poderá ser promovido se reduzir revisão sem perda relevante de cobertura em validação. |
| Próximo ciclo | Evidências de chute e movimento de rede | Evidências complementares, avaliadas uma por vez | Não serão transformadas em regra de confirmação automática. |
| Marco final | Avaliação única | Protocolo e configuração congelados aplicados ao TEST | Nenhuma decisão de ajuste será tomada a partir do TEST. |

## Fluxo operacional atual

```text
vídeo
  -> mineração de coocorrências bola--estrutura
  -> agrupamento e ranking temporal
  -> verificador visual e fusão congelados
  -> seleção do modo operacional
  -> união limitada de segmentos
  -> clipes com evidência visual
  -> revisão e tag humana
```

## Próximo experimento: critério de promoção

A trajetória local será executada em paralelo ao V4. Para ser integrada ao
fluxo principal, deverá ter protocolo registrado antes da execução, preservar
a cobertura no conjunto de validação e demonstrar redução mensurável do tempo
ou da quantidade de clipes a revisar. Enquanto isso, sua saída é diagnóstica e
nunca altera a etiqueta automática do segmento.
