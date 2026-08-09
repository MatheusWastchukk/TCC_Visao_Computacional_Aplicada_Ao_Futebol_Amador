# Guia de preenchimento dos CSVs do piloto

## 1. `matches.csv`

Cada linha representa uma partida completa.

Exemplo:

```csv
match_id,file_name,duration,fps,resolution,file_size,stadium,camera_position,lighting,goal_color,other_goal_structures,notes
match_01,partida_01.mp4,01:42:15,30,1920x1080,1.6GB,campo_a,arquibancada_lateral,diurna,branca,nao,camera acompanha a bola e nao ha replay
```

Campos:

- `match_id`: identificador curto e fixo, como `match_01`.
- `file_name`: nome do arquivo de video.
- `duration`: duracao no formato `HH:MM:SS`.
- `fps`: quadros por segundo, como `30` ou `60`.
- `resolution`: largura x altura, como `1920x1080`.
- `file_size`: tamanho aproximado, como `1.6GB`.
- `stadium`: identificador do campo/estadio, sem precisar revelar nome real.
- `camera_position`: exemplo `arquibancada_lateral`, `lateral_alta`, `central`.
- `lighting`: exemplo `diurna`, `noturna`, `sombra`, `mista`.
- `goal_color`: exemplo `branca`, `amarela`, `metalica`.
- `other_goal_structures`: use `sim`, `nao` ou `incerto`.
- `notes`: observacoes livres.

## 2. `timestamps_template.csv`

Cada linha representa um frame que voce quer extrair de uma partida.

Exemplo:

```csv
match_id,video_path,timestamp_seconds,label
match_01,data/raw_matches/partida_01.mp4,120.500,main_goal
match_01,data/raw_matches/partida_01.mp4,300.000,not_goal
```

Campos:

- `match_id`: precisa bater com o `match_id` do `matches.csv`.
- `video_path`: caminho local do video dentro do projeto ou caminho absoluto.
- `timestamp_seconds`: tempo em segundos. Exemplo: `02:00.500` vira `120.500`.
- `label`: classe esperada no frame.

Labels recomendados:

- `main_goal`: gol principal visivel.
- `secondary_goal`: trave secundaria, goleira pequena ou estrutura parecida.
- `not_goal`: frame sem estrutura de gol principal.

## Como escolher timestamps

Assista ao video e anote tempos em tres grupos:

1. Positivos faceis: gol principal grande, claro e quase inteiro.
2. Positivos dificeis: gol pequeno, parcial, com jogadores na frente, sombra ou camera em movimento.
3. Negativos dificeis: linhas do campo, uniformes brancos, placas, alambrados, traves secundarias e meio-campo.

Para converter tempo de video em segundos:

```text
MM:SS = minutos * 60 + segundos
HH:MM:SS = horas * 3600 + minutos * 60 + segundos
```

Exemplos:

```text
00:02:00.500 -> 120.500
00:10:45.250 -> 645.250
01:05:30.000 -> 3930.000
```

## Primeiro lote recomendado

Antes de extrair 900 frames, faca um lote pequeno:

- 10 `main_goal` por partida;
- 10 `not_goal` por partida;
- 3 a 5 `secondary_goal`, se existirem.

Se esse lote pequeno estiver organizado e abrindo corretamente, aumente para a meta do piloto:

- 150 positivos por partida;
- 150 negativos por partida.
