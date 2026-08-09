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

A regiao deve representar o espaco visual aproximado do interior do gol no plano da imagem. Para a Abordagem A, essa regiao sera tratada como um poligono em coordenadas de pixel. A unidade principal de analise sera um clipe curto de lance, normalmente com 10 a 15 segundos, contendo um evento de gol ou nao gol. Mesmo nesses clipes, a marcacao nao precisa valer desde o primeiro frame: o lance pode comecar com a camera no meio do campo e o gol pode aparecer apenas apos 3 a 5 segundos. Portanto, a regiao do gol deve ser associada a uma cena continua ou a um intervalo de frames no qual o gol esteja visivel e a camera permaneca suficientemente coerente.

Formato conceitual:

```json
{
  "video_id": "Video_Gol_1",
  "scene_id": "S001",
  "start_frame": 0,
  "end_frame": 360,
  "goal_visible_from": 105,
  "goal_visible_to": 318,
  "frame_reference": 142,
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

## Cenas e continuidade da camera

Antes de avaliar um lance, o video deve ser tratado como uma sequencia de cenas continuas. Uma nova cena deve ser iniciada quando houver:

- corte de camera;
- mudanca brusca de enquadramento;
- desaparecimento do gol;
- movimento rapido demais da camera;
- perda de confianca no rastreamento da regiao do gol.

Depois de um corte ou de uma perda forte de continuidade, qualquer posicao anterior do gol deve ser descartada. A regiao do gol so pode ser reutilizada enquanto a cena atual preservar continuidade visual suficiente.

Em clipes curtos de 10 a 15 segundos, a existencia de frames iniciais sem gol visivel nao invalida o lance. Nesses frames, a bola pode ser detectada e rastreada normalmente, mas a relacao entre bola e regiao do gol fica indisponivel ate que uma regiao valida seja marcada ou carregada. O evento so pode ser decidido como `possivel_gol` ou `nao_gol` quando o gol estiver visivel antes ou durante o momento decisivo; caso o gol apareca tarde demais para avaliar a entrada da bola, o resultado deve ser `inconclusivo`.

Estados conceituais para o controle da regiao:

| Estado | Significado | Proxima acao |
|---|---|---|
| `gol_nao_visivel` | A estrutura ou regiao aproximada do gol nao esta disponivel no quadro. | Aguardar ate o gol aparecer. |
| `aguardando_marcacao` | O gol aparece, mas ainda nao existe poligono confiavel para a cena. | Marcar manualmente ou carregar configuracao da cena. |
| `regiao_inicializada` | Existe um frame de referencia com poligono marcado para o gol. | Propagar a regiao para frente e, quando util, para tras. |
| `rastreando_gol` | Ha poligono valido e continuidade visual suficiente. | Propagar a regiao e avaliar a bola. |
| `rastreamento_perdido` | A regiao deixou de ser confiavel. | Interromper a avaliacao do evento nessa cena. |
| `aguardando_nova_marcacao` | O gol reapareceu apos corte ou perda. | Solicitar ou carregar nova marcacao. |

## Propagacao da regiao do gol

Quando a camera se movimentar dentro da mesma cena, o poligono do gol nao deve permanecer fixo no frame inicial. A versao semiautomatica deve tentar acompanhar a regiao usando pontos visuais estaticos proximos ao gol, como traves, linhas do campo, placas, alambrado, gramado e estruturas do fundo.

Na implementacao, uma estrategia viavel e usar fluxo optico Lucas-Kanade com OpenCV para rastrear pontos entre frames consecutivos e estimar uma transformacao entre o frame anterior e o atual. Em seguida, essa transformacao pode ser aplicada aos vertices do poligono do gol.

Para clipes curtos, a marcacao pode ser feita em um frame de referencia no qual o gol esteja mais nitido, mesmo que esse frame esteja no meio do lance. A partir desse frame, o rastreamento pode ser executado em dois sentidos:

- para frente, ate o fim do lance ou ate a perda de confianca;
- para tras, ate o primeiro frame em que o gol aparece de forma suficiente ou ate a perda de confianca.

Essa estrategia permite que um clipe comece no meio-campo, registre o historico inicial da bola e so ative a regiao do gol quando ela estiver visualmente disponivel.

Pontos localizados em jogadores devem ser evitados, pois jogadores possuem movimento proprio e podem induzir deslocamento falso da camera. Quando houver deteccao de jogadores disponivel, suas caixas devem ser usadas como mascara de exclusao para a busca de pontos de rastreamento.

O rastreamento da regiao deve ser interrompido quando ocorrer qualquer uma das condicoes abaixo:

- poucos pontos visuais validos permanecem;
- erro da transformacao estimada fica alto;
- o poligono muda de area ou proporcao de forma incoerente;
- o poligono sai do frame;
- a regiao se deforma excessivamente;
- ocorre corte de camera;
- o gol desaparece ou deixa de ser reconhecivel.

Quando o rastreamento da regiao for perdido no momento critico do lance, o resultado deve ser `inconclusivo`, nao `nao_gol`.

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
| GOL-A-007 | Regiao confiavel na cena | A regiao do gol esta valida para a cena atual, seja por marcacao manual, configuracao carregada ou propagacao rastreada com confianca suficiente. |
| GOL-A-008 | Sobreposicao temporal util | Existe intersecao temporal entre a trajetoria decisiva da bola e o intervalo em que a regiao do gol esta visivel/confiavel. |

Se GOL-A-001, GOL-A-007 ou GOL-A-008 nao forem atendidas, o resultado deve ser `inconclusivo`, nao `nao_gol`.

## Tratamento de situacoes especiais

| Situacao | Decisao |
|---|---|
| Estrutura do gol parcialmente visivel | Aceitar se a regiao interna puder ser marcada com confianca. |
| Estrutura do gol invisivel | Classificar como `inconclusivo`, salvo se houver anotacao manual confiavel da regiao. |
| Gol ausente no inicio do clipe | Rastrear a bola normalmente e aguardar a regiao do gol ficar disponivel; nao classificar antes disso. |
| Gol aparece somente depois do momento decisivo | Classificar como `inconclusivo`, pois a entrada nao pode ser verificada. |
| Bola na rede pelo lado de fora | Classificar como `nao_gol` se a trajetoria nao entrar pela regiao interna; se a distincao nao for visivel, `inconclusivo`. |
| Perda de deteccao antes da regiao | Permitir lacuna de ate 5 frames; acima disso, reduzir confianca ou marcar `inconclusivo`. |
| Corte de camera no momento do chute ou entrada | Marcar `inconclusivo`, pois a trajetoria nao pode ser verificada. |
| Corte de camera antes do lance terminar | Encerrar a cena atual; se a entrada no gol nao foi verificada antes do corte, marcar `inconclusivo`. |
| Rastreamento da regiao do gol perdido | Marcar `inconclusivo` no trecho afetado e exigir nova marcacao ou nova configuracao. |
| Bola pequena ou borrada | Aceitar somente se houver consistencia temporal; deteccoes isoladas nao bastam. |
| Jogador ou trave confundido com bola | Classificar como `nao_gol` se a trajetoria for inconsistente. |
| Rebote dentro da area sem entrada no gol | Classificar como `nao_gol`. |
| Camera tremida | Manter classificacao somente se a regiao e a trajetoria continuarem confiaveis. |

## Confianca inicial

A Abordagem A podera retornar uma confianca heuristica entre 0 e 1. A confianca nao sera probabilidade estatistica; sera um indice de forca da evidencia.

Sugestao inicial:

| Evidencia | Peso |
|---|---:|
| Bola detectada antes da entrada | 0.15 |
| Trajetoria compativel com o gol | 0.20 |
| Entrada no poligono do gol | 0.30 |
| Confirmacao temporal ou sumico dentro da regiao | 0.15 |
| Ausencia de corte/oclusao critica | 0.10 |
| Regiao do gol rastreada com confianca na cena | 0.10 |

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
| O clipe comeca no meio-campo e o gol so aparece depois que a bola ja deveria ter entrado ou saido. | `inconclusivo` | Falta regiao do gol no periodo decisivo. |
| O gol aparece antes do chute, mas o rastreamento da regiao se perde antes da bola chegar. | `inconclusivo` | A regiao nao esta confiavel no momento critico. |

## Checklist manual de classificacao

Antes de classificar um lance, preencher:

| Pergunta | Resposta esperada |
|---|---|
| A fonte do video foi registrada? | Sim/nao |
| O clipe possui 10 a 15 segundos ou duracao equivalente de lance? | Sim/nao |
| O gol ou sua regiao aproximada esta visivel? | Sim/nao/parcial |
| Em qual frame/tempo o gol comeca a aparecer? | Numero/incerto |
| Qual frame foi usado como referencia da marcacao? | Numero/incerto |
| A regiao do gol pode ser marcada com confianca? | Sim/nao |
| O trecho pertence a uma cena continua sem corte critico? | Sim/nao/incerto |
| A regiao do gol permaneceu confiavel ate o momento critico? | Sim/nao/incerto |
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
- A propagacao da regiao do gol pode acumular erro ao longo da cena.
- Cortes, zoom e movimentos rapidos da camera exigem nova marcacao ou retorno `inconclusivo`.
- Oclusoes, baixa resolucao e bola muito pequena podem gerar casos inconclusivos.
- O sistema pode detectar `possivel_gol`, mas a decisao oficial continuaria fora do escopo.
- A calibracao automatica completa do campo e da camera fica fora da implementacao inicial e deve ser tratada como trabalho futuro.

## Taxa de lances avaliaveis

Alem das metricas de acerto entre `possivel_gol` e `nao_gol`, a avaliacao deve registrar quantos clipes foram classificados como `inconclusivo`. Para o TCC, esses casos representam lances nao avaliaveis pela abordagem atual e ajudam a medir a viabilidade tecnica em videos amadores.

Formato conceitual de resumo:

| Resultado de avaliacao | Quantidade |
|---|---:|
| Gol corretamente identificado | A preencher |
| Nao gol corretamente identificado | A preencher |
| Classificacao incorreta | A preencher |
| Lance inconclusivo/nao avaliavel | A preencher |

## Decisao para implementacao inicial

Para o primeiro prototipo, a Abordagem A deve usar:

- clipes curtos de lance, preferencialmente com 10 a 15 segundos;
- divisao conceitual do video em cenas continuas;
- rastreamento da bola desde o inicio do clipe, mesmo quando o gol ainda nao estiver visivel;
- regiao do gol marcada manualmente ou carregada por configuracao para a cena atual;
- frame de referencia em que o gol esteja claramente visivel;
- intervalo de visibilidade da regiao, com `goal_visible_from` e `goal_visible_to` quando possivel;
- descarte da regiao anterior apos corte de camera ou perda de continuidade;
- propagacao semiautomatica do poligono para frente e, quando util, para tras a partir do frame de referencia;
- interrupcao da avaliacao quando o rastreamento da regiao deixar de ser confiavel;
- centro da caixa da bola como ponto de referencia;
- pelo menos 2 frames consecutivos dentro da regiao;
- tolerancia de ate 5 frames sem bola;
- janela de 30 frames antes e 30 frames depois;
- saida padronizada com `possivel_gol`, `nao_gol` ou `inconclusivo`;
- confianca heuristica registrada junto ao frame de evidencia.
