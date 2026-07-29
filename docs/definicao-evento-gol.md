# Definicao Formal do Evento de Gol

Data de criacao: 29/07/2026

## Objetivo

Este documento define o que o prototipo considerara como evento candidato a gol. A definicao e operacional: deve orientar anotacao manual, implementacao da Abordagem A, comparacao da Abordagem B e avaliacao dos resultados.

O sistema nao substitui arbitragem, VAR ou tecnologia oficial de linha do gol. O resultado sera tratado como evidencia computacional de **possivel gol**.

## Definicao principal

Um candidato a gol ocorre quando a trajetoria estimada da bola entra em uma regiao previamente definida como interior do gol, obedecendo a regras temporais e direcionais. O resultado e tratado como evento provavel, nao como comprovacao oficial de ultrapassagem total da linha.

## Saidas possiveis

| Classe | Significado | Uso esperado |
|---|---|---|
| `possivel_gol` | Ha evidencia espacial e temporal suficiente de que a bola entrou na regiao definida como interior do gol. | Evento positivo para a Abordagem A. |
| `nao_gol` | A bola nao entra na regiao do gol, ou a evidencia contradiz a entrada. | Evento negativo. |
| `inconclusivo` | A bola some, ha corte de camera, o gol esta fora do quadro ou a evidencia visual e insuficiente. | Caso ambiguo, analisado separadamente. |

## Regiao do gol

Na versao inicial, a regiao do gol sera marcada manualmente ou carregada de um arquivo de configuracao. Essa decisao reduz o risco tecnico de tentar detectar automaticamente a estrutura do gol antes de existir um pipeline confiavel.

A regiao deve representar o espaco visual aproximado do interior do gol no plano da imagem. Para a Abordagem A, essa regiao sera tratada como um poligono em coordenadas de pixel.

Formato conceitual:

```json
{
  "video_id": "Video_Gol_1",
  "frame_reference": 120,
  "image_width": 1920,
  "image_height": 1080,
  "goal_region": [
    [1500, 420],
    [1840, 430],
    [1850, 720],
    [1490, 700]
  ]
}
```

## Janela temporal

A analise inicial considerara uma janela local em torno da entrada da bola na regiao do gol.

Parametros iniciais recomendados:

| Parametro | Valor inicial | Justificativa |
|---|---:|---|
| Frames consecutivos dentro da regiao | 2 | Reduz falso positivo de uma unica deteccao instavel. |
| Tolerancia de perda da bola | 5 frames | Permite pequenas oclusoes ou falhas do detector. |
| Janela antes do evento | 30 frames | Ajuda a verificar direcao da trajetoria. |
| Janela depois do evento | 30 frames | Ajuda a verificar permanencia, sumico ou continuidade. |
| Distancia minima de deslocamento | A calibrar | Evita classificar bola parada ou ruido como evento. |

Esses valores sao parametros iniciais e devem ser ajustados apenas no conjunto de desenvolvimento, antes do congelamento do teste final.

## Regra computacional da Abordagem A

Um lance sera classificado como `possivel_gol` quando todas as condicoes abaixo forem atendidas:

| ID | Condicao | Descricao |
|---|---|---|
| GOL-A-001 | Regiao definida | Existe uma regiao do gol valida para o clipe. |
| GOL-A-002 | Bola rastreada antes do evento | Ha posicao estimada da bola em frames anteriores suficientes para inferir trajetoria. |
| GOL-A-003 | Entrada na regiao | O centro da bola ou a maior parte da caixa da bola entra no poligono do gol. |
| GOL-A-004 | Confirmacao temporal | A bola permanece dentro da regiao por pelo menos 2 frames consecutivos, ou entra e some dentro de uma tolerancia de ate 5 frames. |
| GOL-A-005 | Direcao compativel | A trajetoria antes da entrada aponta para a regiao do gol, evitando casos de deteccao isolada. |
| GOL-A-006 | Sem contradicao visual clara | Nao ha evidencia forte de que a bola passou por fora da rede ou que houve corte de camera no momento decisivo. |

Se GOL-A-001 nao for atendida, o resultado deve ser `inconclusivo`, nao `nao_gol`.

## Tratamento de situacoes especiais

| Situacao | Decisao |
|---|---|
| Estrutura do gol parcialmente visivel | Aceitar se a regiao interna puder ser marcada com confianca. |
| Estrutura do gol invisivel | Classificar como `inconclusivo`, salvo se houver anotacao manual confiavel da regiao. |
| Bola na rede pelo lado de fora | Classificar como `nao_gol` se a trajetoria nao entrar pela regiao interna; se a distincao nao for visivel, `inconclusivo`. |
| Perda de deteccao antes da regiao | Permitir lacuna de ate 5 frames; acima disso, reduzir confianca ou marcar `inconclusivo`. |
| Corte de camera no momento do chute ou entrada | Marcar `inconclusivo`, pois a trajetoria nao pode ser verificada. |
| Bola pequena ou borrada | Aceitar somente se houver consistencia temporal; deteccoes isoladas nao bastam. |
| Jogador ou trave confundido com bola | Classificar como `nao_gol` se a trajetoria for inconsistente. |
| Rebote dentro da area sem entrada no gol | Classificar como `nao_gol`. |
| Camera tremida | Manter classificacao somente se a regiao e a trajetoria continuarem confiaveis. |

## Confianca inicial

A Abordagem A podera retornar uma confianca heuristica entre 0 e 1. A confianca nao sera probabilidade estatistica; sera um indice de forca da evidencia.

Sugestao inicial:

| Evidencia | Peso |
|---|---:|
| Bola detectada antes da entrada | 0.20 |
| Trajetoria compativel com o gol | 0.25 |
| Entrada no poligono do gol | 0.30 |
| Confirmacao temporal ou sumico dentro da regiao | 0.15 |
| Ausencia de corte/oclusao critica | 0.10 |

Interpretacao:

| Faixa | Resultado |
|---|---|
| `>= 0.70` | `possivel_gol` |
| `0.40` a `0.69` | `inconclusivo` |
| `< 0.40` | `nao_gol` |

Esses limiares devem ser tratados como configuracao experimental.

## Exemplos positivos

| Exemplo | Classificacao | Justificativa |
|---|---|---|
| Bola chutada em direcao ao gol, entra no poligono e permanece por 2 ou mais frames. | `possivel_gol` | Ha entrada espacial e confirmacao temporal. |
| Bola entra no poligono e desaparece dentro da regiao por ate 5 frames, sem corte de camera. | `possivel_gol` | O sumico e compativel com oclusao/rede apos entrada. |
| Bola cruza em direcao ao gol e aparece no interior da meta apos pequena falha de tracking. | `possivel_gol` | A tolerancia cobre pequena lacuna. |

## Exemplos negativos

| Exemplo | Classificacao | Justificativa |
|---|---|---|
| Chute passa ao lado da trave e nunca entra no poligono. | `nao_gol` | Nao ha entrada na regiao do gol. |
| Bola toca a rede pelo lado de fora e a trajetoria passa fora da regiao interna. | `nao_gol` | Ha evidencia de lado externo. |
| Objeto branco perto do gol e detectado como bola, mas nao tem trajetoria compativel. | `nao_gol` | Falha isolada de deteccao nao atende a regra temporal. |

## Exemplos ambiguos

| Exemplo | Classificacao | Justificativa |
|---|---|---|
| Camera corta no momento exato em que a bola se aproxima do gol. | `inconclusivo` | Nao e possivel verificar entrada. |
| Gol esta parcialmente fora do enquadramento e a regiao interna nao pode ser marcada. | `inconclusivo` | Falta referencia espacial confiavel. |
| Bola some atras de jogadores antes de chegar ao gol e reaparece depois da comemoracao. | `inconclusivo` | A trajetoria decisiva nao foi observada. |

## Checklist manual de classificacao

Antes de classificar um lance, preencher:

| Pergunta | Resposta esperada |
|---|---|
| A fonte do video foi registrada? | Sim/nao |
| O gol ou sua regiao aproximada esta visivel? | Sim/nao/parcial |
| A regiao do gol pode ser marcada com confianca? | Sim/nao |
| A bola aparece antes do evento? | Sim/nao/parcial |
| A trajetoria da bola aponta para o gol? | Sim/nao/incerto |
| A bola entra na regiao do gol? | Sim/nao/incerto |
| Ha corte de camera no momento critico? | Sim/nao |
| Ha oclusao severa no momento critico? | Sim/nao |
| Classificacao final | `possivel_gol`/`nao_gol`/`inconclusivo` |
| Observacao | Texto curto |

## Tabela de teste manual

A tabela abaixo deve ser usada para validar se a definicao e compreensivel por outra pessoa. O criterio de aceite final da TCC-007 exige 10 lances classificados manualmente e revisados.

| ID | Clipe | Tipo esperado pelo nome | Regiao do gol visivel? | Bola visivel no evento? | Classificacao manual | Revisao independente | Observacao |
|---|---|---|---|---|---|---|---|
| L001 | `Video_Gol_1.mp4` | Gol | A revisar | A revisar | A revisar | Pendente | Primeiro clipe positivo disponivel. |
| L002 | `Video_Nao_Gol_1.mp4` | Nao gol | A revisar | A revisar | A revisar | Pendente | Clipe negativo disponivel. |
| L003 | `Video_Nao_Gol_2.mp4` | Nao gol | A revisar | A revisar | A revisar | Pendente | Clipe negativo disponivel. |
| L004 | `Video_Nao_Gol_3.mp4` | Nao gol | A revisar | A revisar | A revisar | Pendente | Clipe negativo disponivel. |
| L005 | `Video_Nao_Gol_4.mp4` | Nao gol | A revisar | A revisar | A revisar | Pendente | Clipe negativo disponivel. |
| L006 | A selecionar | A definir | A revisar | A revisar | A revisar | Pendente | Necessario completar amostra de 10 lances. |
| L007 | A selecionar | A definir | A revisar | A revisar | A revisar | Pendente | Necessario completar amostra de 10 lances. |
| L008 | A selecionar | A definir | A revisar | A revisar | A revisar | Pendente | Necessario completar amostra de 10 lances. |
| L009 | A selecionar | A definir | A revisar | A revisar | A revisar | Pendente | Necessario completar amostra de 10 lances. |
| L010 | A selecionar | A definir | A revisar | A revisar | A revisar | Pendente | Necessario completar amostra de 10 lances. |

## Limitacoes

- A definicao usa coordenadas de imagem, nao coordenadas reais do campo.
- A entrada no poligono do gol e aproximada e depende da camera.
- A regra nao comprova ultrapassagem total da linha.
- A regiao manual pode introduzir variacao entre anotadores.
- Oclusoes, baixa resolucao e bola muito pequena podem gerar casos inconclusivos.
- O sistema pode detectar `possivel_gol`, mas a decisao oficial continuaria fora do escopo.

## Decisao para implementacao inicial

Para o primeiro prototipo, a Abordagem A deve usar:

- regiao do gol marcada manualmente;
- centro da caixa da bola como ponto de referencia;
- pelo menos 2 frames consecutivos dentro da regiao;
- tolerancia de ate 5 frames sem bola;
- janela de 30 frames antes e 30 frames depois;
- saida padronizada com `possivel_gol`, `nao_gol` ou `inconclusivo`;
- confianca heuristica registrada junto ao frame de evidencia.
