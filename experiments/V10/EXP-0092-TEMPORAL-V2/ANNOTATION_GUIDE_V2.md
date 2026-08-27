# Guia de anotação — EXP-0092-TEMPORAL-V2

## Objetivo

Anotar 8 blocos contínuos de 5 minutos, um em cada partida de validação. O objetivo desta etapa é medir a persistência da localização `goal_structure` em vídeo contínuo, incluindo interrupções e mudanças de câmera. Esta etapa ainda não valida o evento de gol.

## Arquivos

- `continuous_blocks_v2.csv`: blocos prontos para revisar no vídeo real.
- `interval_annotations_template_v2.csv`: uma linha para cada intervalo homogêneo dentro de um bloco.

## Como revisar

1. Abra o vídeo indicado em `continuous_blocks_v2.csv`.
2. Vá ao `start_time_hhmmss` e assista até o `end_time_hhmmss`.
3. Divida o bloco sempre que o estado mudar. Não é necessário criar uma linha a cada frame.
4. Preencha `start_seconds` e `end_seconds` em segundos relativos ao vídeo e também os campos `hhmmss`.
5. Use os rótulos abaixo.

## Rótulos

- `goal_structure`: pelo menos uma estrutura de gol está visível e avaliável.
- `not_goal`: a cena está visível, mas não há estrutura de gol relevante.
- `inconclusive`: a cena não permite decidir com segurança, por exemplo durante uma transição.
- `not_evaluable`: a estrutura está fora do campo, encoberta ou a imagem não permite avaliação.

Em `structure_scope`, registre `main`, `secondary`, `multiple` ou `unknown`. Uma estrutura secundária continua sendo `goal_structure`; não transformar falta de enquadramento do gol principal em `not_goal`.

## Critérios de anotação

- Inclua intervalos consecutivos, sem lacunas, cobrindo todo o bloco.
- Não classifique o bloco inteiro por causa de um único frame.
- Registre em `notes` panorâmica, zoom, troca de câmera, overlay, gol parcial, jogada parada ou outra causa de interrupção.
- Se houver dúvida real, use `inconclusive`; se não houver visibilidade suficiente para avaliação, use `not_evaluable`.
- Depois de preencher, altere `annotation_status` do bloco para `completed`.

## Exemplo

```csv
temporal_v2_interval_001,temporal_v2_block_001,match_03,Match_03_A.mp4,1864.000,1878.500,00:31:04,00:31:18,goal_structure,main,completed,Trave visivel; camera estavel.
```

Os parâmetros iniciais do experimento ficam fixos: confiança YOLO `0.25`, amostragem `5 FPS` e limiar de persistência/aparência `70%`. Eles só devem ser alterados em uma versão experimental posterior.

## Anotador visual

Na raiz do projeto, execute no PowerShell:

```powershell
.\.venv\Scripts\python.exe experiments\temporal\annotate_temporal_blocks_v2.py `
  --blocks experiments\V10\EXP-0092-TEMPORAL-V2\continuous_blocks_v2.csv `
  --output experiments\V10\EXP-0092-TEMPORAL-V2\interval_annotations_v2.csv
```

Controles principais:

- `Espaço`: reproduzir ou pausar;
- `←` e `→`: voltar ou avançar 1 segundo;
- `↑` e `↓`: avançar ou voltar 5 segundos;
- `1` a `4`: selecionar o rótulo;
- `M`, `R`, `T`: estrutura principal, secundária ou múltiplas;
- `[`: marcar início do intervalo;
- `]`: marcar fim e salvar o intervalo;
- `N` e `P`: próximo ou bloco anterior;
- `S`: salvar progresso; `Q`: sair salvando.

O aplicativo grava os tempos em segundos e em `hh:mm:ss`, além de atualizar automaticamente o status dos blocos.

### Modo automático para mudanças rápidas

O aplicativo inicia no modo automático. Nesse modo, não é necessário usar `[` e `]` a cada mudança:

1. Pressione `1`, `2`, `3` ou `4` para iniciar o estado atual.
2. Deixe o vídeo reproduzir.
3. Quando a situação mudar, pressione o novo rótulo.
4. O aplicativo fecha automaticamente o intervalo anterior e inicia o novo no instante atual.
5. Ao chegar ao fim do bloco, pressione `E` para fechar o intervalo final.

Use `A` para alternar entre o modo automático e o modo manual. No modo manual, continuam valendo `[` para início e `]` para fim.

Para precisão de frame, pause e use `,` e `.` para voltar ou avançar um frame. As setas continuam avançando 1 ou 5 segundos.

Pressione e solte `Espaço` uma única vez para reproduzir ou pausar. O aplicativo agora ignora repetições geradas enquanto a tecla permanece pressionada.
