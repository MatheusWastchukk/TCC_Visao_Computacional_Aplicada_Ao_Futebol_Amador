# Auditoria da qualidade das propostas BALL

## O que revisar

Em cada frame, as caixas `BALL #n` são propostas do detector e aparecem em ordem de confiança. A caixa verde é a anotação manual da bola verdadeira.

- Se conseguir identificar a bola, arraste uma caixa apertada em volta dela e pressione **S**. Faça isso mesmo quando nenhuma caixa do detector coincide com a bola.
- Se a bola não estiver visível no frame, pressione **N**. Caixas `BALL` ainda podem aparecer no overlay; elas continuam sendo propostas do detector.
- Se não conseguir decidir, pressione **U**.
- Opcional: pressione `1` chão, `2` aérea, `3` oclusão, `4` pós-chute, `5` perto do gol/goleiro, `6` câmera em movimento, `7` cruzamento, `8` rebote, `9` múltiplas propostas. As tags alternam ligadas/desligadas.
- **Espaço** pula sem concluir; `A`/`D` voltam/avançam entre pendentes; `Q` salva e sai. O arquivo de revisão é gravado atomicamente após cada decisão e a execução retoma nos pendentes.

Os 400 frames são todos TRAIN: 144 amostras dos 12 misses, 144 de 24 gols fáceis e 112 controles candidatos com rótulo fraco. Os controles fracos não são gols negativos confirmados. Nos misses e controles fáceis há pequenos blocos de três frames consecutivos amostrados a cerca de 5 FPS para medir presença de proposta ao longo do tempo.

## GPU e modelos

O fluxo real completou na AMD Radeon RX 9060 XT com PyTorch `2.9.1+rocm7.2.1`, `cuda:0`, pesos congelados `EXP-0092` (GOAL) e `BALL-EXP-0005` (BALL), limiares 0,25/0,05 e tamanhos 640/960. Uma mensagem de parsing de `offload-arch` apareceu na inicialização, mas não impediu o smoke nem a inferência completa de 400 frames. Cada frame foi pontuado mesmo sem proposta GOAL ou BALL.

## Comandos

Da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe app\annotate_ball_proposal_quality.py
```

O progresso fica em `run_v2/manual_ball_review_v1.csv`. Após completar todos os frames:

```powershell
.\.venv\Scripts\python.exe experiments\ball\evaluate_ball_proposal_quality_audit_v1.py
```

O avaliador mede recall da proposta da bola em IoU 0,3 e 0,5, número de propostas falsas por frame, rank de confiança da proposta correta e cobertura em pares consecutivos. Frames `UNCERTAIN` não entram nos denominadores. As métricas são amostrais, limitadas às condições e frames selecionados.
