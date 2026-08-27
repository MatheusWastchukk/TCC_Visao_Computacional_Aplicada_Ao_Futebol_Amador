# Organização dos experimentos por versão

## Estrutura

Cada versão metodológica possui uma pasta própria em `experiments/` e cada experimento fica dentro da versão correspondente.

```text
experiments/
├── baselines/EXP-0001/
├── V1/EXP-0013/
├── V2/EXP-0014/
├── V3/EXP-0015/
├── V4/EXP-0016/
├── V5/EXP-0017/
├── V6/EXP-0018/
├── V7/EXP-0019/
├── V8/EXP-0020/
├── V8.5/EXP-0021 ... EXP-0029/
├── V8.6/EXP-0030 ... EXP-0039/
├── V8.7/EXP-0040 ... EXP-0044/
└── V10/EXP-0045 ... EXP-0092/
```

## Mapeamento

| Pasta | Experimentos | Escopo |
|---|---|---|
| `baselines` | `EXP-0001` | baseline clássico inicial |
| `V1` a `V7` | `EXP-0013` a `EXP-0019` | evolução inicial da geometria |
| `V8` | `EXP-0020` | geometria contínua/soft geometry |
| `V8.5` | `EXP-0021` a `EXP-0029` | geometria aprendida, testes externos e calibração |
| `V8.6` | `EXP-0030` a `EXP-0039` | propostas espaciais, topologia e rankers |
| `V8.7` | `EXP-0040`, `EXP-0042` a `EXP-0044` | crops e classificador híbrido |
| `V10` | `EXP-0045` em diante | detector YOLO, ampliações, validação e piloto temporal |

Os EXP-0041, EXP-0046 e EXP-0048 aparecem em referências históricas, mas não possuem pasta de artefatos no checkout atual.
