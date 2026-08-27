# Guia de anotaÃ§Ã£o temporal V10

O arquivo `temporal_v10_candidates_v1.csv` contÃ©m 15 trechos contÃ­nuos de 4 segundos, com 8 candidatos positivos e 7 negativos.

Use as colunas `start_time_hhmmss` e `end_time_hhmmss` para localizar o trecho no reprodutor de vÃ­deo. As colunas `start_seconds` e `end_seconds` permanecem no arquivo para uso automÃ¡tico pelo pipeline.

Para cada linha:

1. Abra o vÃ­deo indicado em `video_path`.
2. Verifique o intervalo entre `start_seconds` e `end_seconds`.
3. Ajuste os tempos se o trecho comeÃ§ar antes ou terminar depois da cena Ãºtil.
4. Preencha `true_label`:
   - `goal_structure` se uma ou mais estruturas de gol permanecem visÃ­veis durante o trecho;
   - `not_goal` se nÃ£o hÃ¡ estrutura de gol visÃ­vel durante o trecho;
   - `inconclusive` se houver corte, perda de visibilidade ou dÃºvida relevante.
5. Altere `annotation_status` para `completed`.
6. Explique cortes, estruturas secundÃ¡rias, perda de visibilidade ou qualquer exceÃ§Ã£o em `notes`.

O campo `anchor_label` Ã© apenas uma sugestÃ£o baseada no frame de origem. Ele nÃ£o substitui a conferÃªncia da sequÃªncia contÃ­nua.
