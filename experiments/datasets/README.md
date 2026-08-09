# Anotacoes geometricas

O arquivo `goal_geometry_annotations_v1.csv` contém as 35 estruturas reais do
piloto:

- 30 `main_goal`;
- 5 `secondary_goal`.

Para cada frame, preencher quatro pontos em pixels da imagem original:

```text
LT = topo esquerdo
RT = topo direito
LB = base esquerda
RB = base direita
```

Exemplo:

```text
lt_x,lt_y,rt_x,rt_y,lb_x,lb_y,rb_x,rb_y
420,180,760,185,430,700,750,705
```

Preencher também `annotator`, `annotation_status` e `notes`. Usar
`annotation_status=completed` somente quando os quatro pontos estiverem
revisados. Os frames `secondary_goal` permanecem como estruturas válidas para
a avaliação `goal_structure`.
