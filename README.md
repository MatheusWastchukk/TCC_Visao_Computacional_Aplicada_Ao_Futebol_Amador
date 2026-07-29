# TCC - Deteccao e Analise de Videos de Futebol

Projeto de TCC para deteccao e analise de eventos em videos de futebol usando tecnicas de visao computacional.

## Objetivo

Construir um prototipo capaz de processar trechos de videos de futebol, detectar jogadores e bola, rastrear a trajetoria da bola e identificar candidatos a gol por duas abordagens:

- Abordagem A: heuristica espacial e temporal baseada na trajetoria da bola e na regiao do gol.
- Abordagem B: classificacao temporal ou alternativa hibrida para comparar com a abordagem heuristica.

## Estrutura

```text
app/                 Interface e pontos de entrada da aplicacao
configs/             Arquivos de configuracao
data/                Dados locais nao versionados
docs/                Documentacao do projeto
experiments/         Experimentos e relatorios tecnicos
models/              Pesos e modelos locais nao versionados
notebooks/           Analises exploratorias
outputs/             Frames, videos e relatorios gerados
scripts/             Scripts auxiliares
src/                 Codigo-fonte do pipeline
tests/               Testes automatizados
```

## Ambiente Inicial

Requisitos de sistema:

- Windows 11
- Python 3.11
- Git
- FFmpeg

Dependencias Python iniciais estao em `requirements.txt`.

## Status

- TCC-001: estrutura inicial do projeto criada.
- TCC-002: requisitos funcionais e nao funcionais documentados em `docs/requisitos.md`.
