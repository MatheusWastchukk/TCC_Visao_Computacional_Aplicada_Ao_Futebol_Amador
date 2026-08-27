# Elementos visuais, traves secundarias e evidencias

## Mascara de elementos fixos

Logos, placares, marcas d'agua, textos e placas podem gerar falsos positivos em tecnicas baseadas em limiarizacao, bordas e linhas.

A estrategia inicial deve usar regioes ignoradas configuraveis, aplicadas antes da limiarizacao e da deteccao de linhas. O frame original deve ser preservado para evidencia.

Exemplo:

```yaml
ignored_regions:
  - name: "competition_logo"
    x1: 0
    y1: 0
    x2: 350
    y2: 150
```

Teste recomendado:

1. executar detector sem mascara;
2. medir falsos positivos;
3. aplicar mascara;
4. repetir;
5. comparar precisao, falsos candidatos por hora e tempo.

## Traves secundarias

Videos com traves menores ou goleiras de treinamento nao devem ser descartados automaticamente. Eles fortalecem a avaliacao porque representam falsos positivos reais.

A anotacao deve distinguir:

- `main_goal`;
- `secondary_goal`;
- `not_goal`.

Metricas especificas:

- precisao em cenarios com e sem traves secundarias;
- falsos positivos causados por traves secundarias;
- `secondary_goal_rejection_rate`.

## Evidencias visuais

Para cada evento candidato, o sistema deve poder gerar:

- clipe original recortado;
- clipe anotado;
- frame de evidencia;
- JSON do evento;
- linha em CSV consolidado.

Elementos visuais do clipe:

- caixa ou centro da bola;
- trajetoria recente;
- poligono da regiao do gol;
- timestamp;
- abordagem;
- confianca;
- estado do evento.

A identificacao de jogadores, equipes e time provavel e enriquecimento posterior. O nucleo do TCC continua sendo localizar corretamente o evento e gerar evidencia confiavel.
