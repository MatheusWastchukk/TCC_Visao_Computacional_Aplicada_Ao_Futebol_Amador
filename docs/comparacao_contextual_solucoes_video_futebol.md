# Comparação contextual com soluções de vídeo para futebol

**Consulta:** 20/09/2026.
**Objetivo:** posicionar o projeto por fluxo de trabalho e escopo. Esta não é
uma comparação experimental de acurácia, latência, preço ou cobertura entre
produtos, pois não há protocolo comum, vídeos idênticos nem acesso aos seus
modelos proprietários.

| Sistema | Capacidades declaradas pelo fornecedor | Relação com este projeto |
|---|---|---|
| Veo | Detecta eventos de futebol para navegação, permite filtrar eventos e adicionar eventos ausentes; gera clipes de gols e chutes. | Reforça o valor de separar evento, clipe e correção pelo usuário. O projeto usa candidatos `POSSIBLE_GOAL` em vídeo local, sem alegar equivalência ao produto comercial. |
| Spiideo Play | Gera destaques em clipes, organiza reels e permite editar duração, filtrar, adicionar notas, definir status e exportar o resultado. | É uma referência de interação: lista de clipes, tags, revisão, seleção múltipla e compilação. No projeto, a tag é uma avaliação humana do candidato. |
| Hudl Focus | Automatiza captura e envio de vídeo; disponibiliza ferramentas de análise, revisão, compartilhamento e produtos de dados. | Representa uma plataforma integrada de captura e análise. O escopo do TCC é menor: mineração local de candidatos em um vídeo escolhido pelo usuário. |
| Pixellot | Divulga produção automatizada, gravação, streaming e uso de vídeo para análise esportiva. | É uma referência de ecossistema de produção automatizada, não um comparador direto do minerador de eventos do TCC. |

## Delimitação do projeto

O sistema desenvolvido não substitui plataformas comerciais de captura,
transmissão, gestão de equipes ou estatísticas. Sua contribuição investigada é
um fluxo local e auditável que prioriza trechos candidatos a gol para revisão:

```text
vídeo escolhido pelo usuário
  -> candidatos POSSIBLE_GOAL
  -> clipes com rank e score
  -> tag humana e exportação
```

A comparação será apresentada no artigo como contexto de funcionalidades e
decisões de experiência de uso. Qualquer comparação quantitativa só poderá ser
feita sob protocolo comum e dados equivalentes, o que não é reivindicado aqui.

## Fontes primárias

- Veo, [Football Events](https://updates.veo.co/announcements/new-football-events).
- Spiideo, [AI Highlights and Stories in Spiideo Play](https://support.spiideo.com/en/articles/15070462-ai-highlights-and-stories-in-spiideo-play).
- Spiideo, [Highlight Studio](https://support.spiideo.com/en/articles/15458409-navigating-the-highlight-studio-in-spiideo-play).
- Hudl, [Soccer Video Analysis Software](https://focus.hudl.com/sports/soccer).
- Pixellot, [brochura de solução de produto](https://www.pixellot.tv/wp-content/uploads/2021/01/Brochure-Product-Solution_V03.pdf).
