# Teste-piloto de localizacao automatica das traves

## Objetivo

Antes de migrar todo o TCC para partidas completas, este piloto deve verificar se a localizacao automatica da estrutura do gol por limiarizacao, linhas e geometria e tecnicamente promissora.

O piloto nao precisa reconhecer gols. Ele deve responder:

1. A estrutura do gol pode ser localizada automaticamente?
2. Quantos falsos positivos aparecem?
3. A localizacao permanece estavel por alguns segundos?
4. A tecnica funciona em partidas diferentes?
5. O custo em CPU e aceitavel?
6. A abordagem deve seguir como principal, hibrida ou ser substituida?

## Decisao

O relatorio final deve produzir uma das decisoes:

- `GO`: a abordagem classica e suficiente para iniciar a implementacao completa;
- `GO_HIBRIDO`: a abordagem gera bons candidatos, mas precisa de detector ou filtro complementar;
- `NO_GO`: limiarizacao e geometria nao sustentam o metodo principal.

## Escopo minimo

- Selecionar 3 partidas completas.
- Catalogar metadados em `experiments/goal_detection_pilot/matches.csv`.
- Extrair 150 frames positivos e 150 negativos por partida.
- Manter `match_id` em todos os arquivos.
- Anotar visibilidade e regiao do gol nos positivos.
- Revisar 20% da anotacao em outro dia.

Quantidade alvo:

```text
450 positivos
450 negativos
900 frames
```

## Experimentos obrigatorios

1. Baseline simples por percentual de pixels claros.
2. Comparacao entre cinza, HSV e limiar adaptativo.
3. Morfologia com diferentes operacoes e kernels.
4. Bordas na imagem original e na mascara.
5. Deteccao e classificacao angular de linhas.
6. Agrupamento geometrico de traves e travessao.
7. Pontuacao de confianca.
8. Avaliacao frame a frame.
9. Generalizacao: ajustar na partida 1 e executar sem alteracao nas partidas 2 e 3.
10. Teste temporal em 15 segmentos de 3 a 5 segundos.
11. Falsos positivos em trechos continuos de 10 a 15 minutos por partida.
12. Custo computacional em CPU.

## Metricas

Registrar por partida, e nao apenas como media global:

- precisao;
- revocacao;
- F1;
- acuracia;
- matriz de confusao;
- IoU medio nos positivos;
- falsos candidatos por hora;
- percentual de frames detectados nos segmentos temporais;
- maior sequencia continua;
- perdas temporais;
- FPS de processamento;
- estimativa de tempo para 90 minutos.

## Referencias iniciais de aceite

Esses valores sao metas de engenharia, nao promessas de resultado:

```text
F1 >= 0.70
falsos candidatos <= 10 por hora
deteccao em >= 70% dos frames dos segmentos
IoU medio >= 0.50
```

## Entregaveis

Ao fim do piloto devem existir:

- `matches.csv`;
- amostra de frames;
- anotacoes;
- configuracoes;
- matriz de experimentos;
- metricas por partida;
- galeria de erros;
- teste temporal;
- medicao de desempenho;
- `REPORT.md` com decisao GO, GO_HIBRIDO ou NO_GO.
