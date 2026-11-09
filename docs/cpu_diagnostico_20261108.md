# CPU: correção da telemetria e otimização da leitura de vídeo

O usuário relatou que o pacote 0.4.0 funcionou em outro computador, mas o
indicador de CPU permaneceu em zero. Esclareceu depois que os 149 minutos de
`match_32` eram uma previsão exibida pelo aplicativo, sem duração final informada.
O equipamento informado é um Samsung Book com Core i5-1135G7, 8 GB de RAM,
SSD de 256 GB, Iris Xe e Windows 11. Nenhuma execução de TEST foi feita para
investigar esse relato.

O estimador atual extrapola `tempo_decorrido / percentual_global * 100`.
O percentual do pipeline usa faixas fixas por etapa, que não representam tempos
iguais de processamento; a inicialização dos modelos também afeta as primeiras
amostras. Por isso a previsão pode variar durante a execução. O número exibido
não permite concluir que a análise efetivamente demorou 149 minutos.

## Indicador de CPU

O backend criava um novo `psutil.Process` a cada chamada de `device_metrics`.
`Process.cpu_percent(interval=None)` precisa da leitura anterior do mesmo objeto;
a primeira leitura apenas estabelece a referência e retorna zero. A correção
mantém um objeto por processo, protege as consultas entre threads e reutiliza a
amostra por um segundo, sem bloquear a API para medir CPU.

O percentual exibido é normalizado pela quantidade de processadores lógicos para
representar a fração da capacidade total da máquina, entre 0 e 100%. Mede o
backend; Electron e os subprocessos FFmpeg não estão incluídos. Antes da primeira
amostra válida, a interface mostra um traço. A memória mantém seu significado de
consumo do backend.

Referência: [psutil: medição de CPU por processo](https://psutil.readthedocs.io/stable/#psutil.Process.cpu_percent).

## Leitura de vídeo

A mineração usava `VideoCapture.read()` em todos os quadros e descartava as
imagens que não pertenciam à amostragem de 5 FPS. O aplicativo passa a usar
`grab()` nos quadros e `retrieve()` somente nos índices amostrados. Os quadros
intermediários continuam sendo percorridos; isso não elimina toda a decodificação
de um vídeo comprimido. Evita recuperar/converter as imagens que serão descartadas.

Os modelos, limiares, resoluções e índices de amostragem são mantidos. A CLI
histórica conserva o comportamento anterior por padrão; a opção `--sampled-decode`
ativa a mudança, e o backend do aplicativo a informa explicitamente.

Referência: [OpenCV: VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html).

## Diagnóstico de desempenho

Novas sessões registram `stage_seconds` com tempos de inicialização, mineração,
ranking, união dos segmentos e geração dos clipes. A exportação JSON inclui esses
tempos, o total de processamento e informações do equipamento. A mineração grava
em `analysis/stages/mining/metrics.json`, por partida, `runtime_profile` com tempo
de leitura, inferência GOAL, inferência BALL e quantidade de quadros/chamadas.
`stage_torch_threads` registra as contagens de threads PyTorch observadas na
thread que executa a análise, nos marcadores das etapas. Isso não representa a
quantidade de threads internas do FFmpeg.

Esses dados permitem escolher a próxima otimização no computador de destino.
Otimizações de inferência e troca de runtime precisam ser medidas e verificadas
em TRAIN. Reduzir resolução/FPS ou trocar modelos pode alterar os candidatos e
exige validação própria; essas mudanças não foram adotadas nesta correção.

## Escopo da comparação local

A comparação usa o trecho de 90 segundos já existente de `match_04`, proveniente
de TRAIN. A verificação de leitura percorre as duas rotas em ordem alternada e
compara os bytes dos quadros amostrados por SHA-256. A comparação do pipeline usa
duas execuções por rota, em sequência, até os segmentos, sem codificação dos clipes.
São verificações técnicas locais, sem estimativa de desempenho para outras máquinas.

Artefatos locais: `build/cpu_optimization_20261108/benchmark.json` e logs na mesma pasta.

Resultados das quatro execuções sequenciais:

| Rota | Execução 1 | Execução 2 | Média |
|---|---:|---:|---:|
| Leitura anterior | 45,062 s | 42,618 s | 43,840 s |
| Recuperação apenas dos quadros amostrados | 39,018 s | 38,160 s | 38,589 s |

A redução da média foi de aproximadamente 12,0%. Nas quatro execuções foram
iguais, byte a byte, os CSVs de quadros/eventos, pool pontuado, candidatos
selecionados, segmentos e `results.json`. Foram 451 quadros amostrados, 126 hits,
quatro grupos e um segmento. Isso valida somente o trecho e essa configuração.
Os tempos não incluem gerar clipes H.264, abrir a interface nem instalar o aplicativo.

Na rota otimizada, a mineração levou 30,585–31,424 s. As inferências GOAL e BALL
somaram 28,834–29,680 s; a leitura ficou em 1,654–1,661 s. A inferência é, portanto,
o principal custo restante **nessa amostra local**. Não se extrapolou esse ganho
para a previsão de 149 minutos exibida no outro computador.

## Verificação do pacote 0.4.1

- O autoteste carregou os modelos e executou inferência CPU e FFmpeg com sucesso.
- O painel foi aberto com um perfil isolado e mostrou CPU diferente de zero durante
  uma análise. A captura foi inspecionada em `build/cpu/desktop_smoke_041/interface.png`.
- O teste final, sem compilação/compactação concorrente, levou 44,573 s para o
  trecho autorizado, incluindo geração de um clipe H.264. Registrou pico de 53,6%
  da capacidade total de CPU no backend. Exportou CSV/JSON/HTML e preservou sessão
  e nota depois de reiniciar.
- O JSON exportado separou mineração (34,661 s), ranking (2,574 s), união dos
  segmentos (0,005 s) e clipes (7,329 s). A thread da análise registrou oito threads
  PyTorch na mineração e quatro no ranking, nesse computador local.
- Evidências finais: `build/cpu/smoke_match04_90s_041_final/smoke_result.json`,
  `export.json`, `backend.log` e `build/cpu/validation_summary_041.json`.

Um teste anterior coincidiu com a montagem do instalador e levou 373,762 s.
Esse tempo documenta carga concorrente; não foi usado para calcular o ganho da
otimização. Os testes locais do pacote e a comparação sequencial do pipeline têm
escopos diferentes e não estimam a duração no Samsung Book informado pelo usuário.

A montagem final reaproveitou as bibliotecas e recursos da construção completa:
o executável foi reconstruído após corrigir os marcadores de tempo, testado com
esses mesmos recursos e reempacotado em NSIS. `scripts/build_cpu.ps1 -Installer`
permanece a rota de construção completa a partir do código final.
