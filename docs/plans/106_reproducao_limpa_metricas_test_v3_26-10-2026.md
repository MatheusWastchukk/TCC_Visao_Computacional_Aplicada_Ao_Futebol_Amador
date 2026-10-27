# Reprodução mínima das métricas TEST V3

**Data:** 26/10/2026<br>
**Objetivo:** verificar a rota de reprodução das métricas congeladas, sem executar inferência, abrir vídeos ou ajustar políticas.<br>
**Resultado:** aprovado para a transformação de artefatos de inferência em métricas; o mesmo cálculo também passou no snapshot temporário da árvore-fonte. Não valida a inferência completa nem uma instalação limpa do ambiente Python/ROCm.

## Cópia isolada

Foi montada uma cópia temporária sob `.tmp/clean_metrics_reproduction_20261026/` contendo somente o script agregador, o manifesto e CSV do catálogo congelado, o manifesto de inferência e os três CSVs de inferência necessários. Não foram copiados vídeos, pesos, embeddings, ambientes virtuais ou código da aplicação. O script usa apenas a biblioteca padrão do Python 3.12.

A primeira execução encontrou BOM UTF-8 em `test_catalog_freeze_manifest_v1.json`; a leitura JSON original falhava porque especificava `utf-8`. O agregador foi corrigido para aceitar `utf-8-sig` nos dois manifests JSON. A execução seguinte terminou com `TEST_V3_METRICS_COMPLETE` para 57 timestamps e nove partidas.

## Validação do snapshot da árvore-fonte

Foi gerada uma cópia temporária `.tmp/repo_snapshot_validation_20261026/` a partir do HEAD mais as alterações inventariadas. Ela não contém `.git`, ambientes virtuais, arquivos LaTeX/PDF nem figuras do artigo; dados de vídeo e pesos não estão versionados na base usada. O snapshot tinha 4.099 arquivos e 22,8 MB antes da instalação do desktop.

Na cópia foram executados `npm.cmd ci`, `npm.cmd ls --depth=0`, audit npm (zero vulnerabilidades) e `npm.cmd run build`. A instalação resolveu 104 pacotes sem dependências ausentes e o build Vite 8.3.5 transformou 1.877 módulos. O aviso do npm de que o postinstall de `ffmpeg-static` não está aprovado permaneceu; `ffmpeg.exe` estava presente no pacote copiado.

O agregador também foi executado dentro desse snapshot. As duas CSVs categóricas permaneceram idênticas; seis valores float da CSV por partida diferiram no máximo `1.4210854715202004e-14` e todos os campos semânticos do resumo ficaram dentro de `1e-12`.

## Comparação com os artefatos congelados

- `goal_coverage_test_v1.csv`: idêntico byte a byte.
- `paired_mode_gains_test_v1.csv`: idêntico byte a byte.
- `per_match_test_v1.csv`: todas as categorias e valores discretos idênticos; seis campos numéricos diferiram por ordem de soma em ponto flutuante, com diferenças inferiores a `1e-12`.
- `summary_test_v1.json`: todos os campos semânticos idênticos após excluir os dois caminhos absolutos dos manifests; valores numéricos comparados com tolerância absoluta de `1e-12`.

Contagens reproduzidas: FAST_V3 45/57, BALANCED_V3 50/57 e HIGH_RECALL_V3 54/57. As razões agregadas de vídeo reproduzidas foram 34,679481%, 46,212160% e 63,070447%, respectivamente. A verificação apenas recalcula métricas de candidatos e catálogo previamente congelados; não abre o TEST para seleção ou retuning.

## Entradas e integridade

| Arquivo de entrada | SHA-256 |
|---|---|
| `experiments/ball/evaluate_frozen_v3_test_metrics_v1.py` | `6441a31261c097acd75049ae136a05b0db1a27db7f6841e08872c503578ca6a9` |
| `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/test_catalog_freeze_manifest_v1.json` | `c2d1d289c30dbdf49280e677f48eac63c77c210178a6fd04bdff4b7b2f41e578` |
| `experiments/ball/GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3/goal_catalog_timestamp_manifest_frozen_v1.csv` | `593fabd777d0e0cf99f723d7a3853d7c8c5d08e951ce54224300204fd824f1b4` |
| `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/inference_manifest_test_v1.json` | `d6c61804e89c3adc5d384f3db6a37cf9cddd7f71460c1278c690a2031a4d88d4` |
| `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/ranked_candidates_test_v1.csv` | `4a427701ca1b108cf9c605a272cd78fc32d0de1d99af57c47c8742013d0705e4` |
| `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/selected_candidates_test_v1.csv` | `3b990357748328f65aa1ba3e866ec4bd65d55ccbd91c0c09ede9b84df00be659` |
| `experiments/ball/TEST-V3-FINAL-EVALUATION-20261019/attempt_02/per_match_inference_test_v1.csv` | `0d7b31ecf136c4fb378b8d0d60a936070e94217b22cea0be6ddb7f17d6f926ca` |

## Comando de reprodução

Executar da raiz do repositório com Python 3.12. O diretório de saída precisa estar ausente ou vazio:

```powershell
& .\.venv-rocm\Scripts\python.exe `
  .\experiments\ball\evaluate_frozen_v3_test_metrics_v1.py `
  --inference .\experiments\ball\TEST-V3-FINAL-EVALUATION-20261019\attempt_02 `
  --output .\.tmp\test_metrics_reproduced `
  --catalog-freeze .\experiments\ball\GOAL-CATALOG-TIMESTAMP-ANNOTATION-TEST-V3\test_catalog_freeze_manifest_v1.json
```

## Limites e pendências

- As cópias de validação são temporárias, não um snapshot de entrega versionado.
- As seis diferenças de ponto flutuante não alteram as contagens nem valores reportados após arredondamento; por isso a aceitação usa tolerância explícita de `1e-12`, não igualdade binária para todos os campos.
- Ainda falta instalar a receita Python/ROCm e os grupos opcionais numa cópia limpa; o snapshot validou o desktop e a rota de métricas com Python já instalado.
- Ainda falta decidir quais artefatos TEST derivados podem ser redistribuídos.
- A auditoria de licenças/direitos e o snapshot seletivo continuam pendentes antes de qualquer release.
