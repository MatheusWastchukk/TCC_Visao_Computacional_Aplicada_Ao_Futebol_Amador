# Auditoria preliminar de licenças e direitos do pacote

**Data:** 26/10/2026<br>
**Estado:** inventário técnico preliminar; não é parecer jurídico nem autorização de distribuição.

## Correção técnica em 05/11/2026

A mineração carregada dinamicamente pelo aplicativo V4 importa Ultralytics.
A classificação anterior como dependência apenas experimental estava incorreta.
`requirements-cpu.txt` inclui essa biblioteca; os pacotes CPU gerados localmente
incluem também os modelos e FFmpeg necessários ao funcionamento. Esses pacotes são
preliminares para teste local e não constituem autorização de redistribuição.
A revisão dos avisos/licenças e da origem dos pesos continua necessária para a entrega.

## Achados no checkout (inventário original de outubro)

- Não foram encontrados `LICENSE`, `COPYING` ou `NOTICE` na raiz ou entre os arquivos rastreados do projeto. O repositório, portanto, não declara aqui uma licença para o código autoral do TCC.
- Os vídeos, clipes, anotações-fonte, pesos/checkpoints e dados externos ficam excluídos do pacote por padrão. A origem, licença e autorização de redistribuição não estão comprovadas de forma suficiente no checkout.
- O catálogo de timestamps TEST e os CSVs de candidatos/resultados são derivados desses dados. Mesmo sem vídeo, a permissão para redistribuir tais derivados precisa ser confirmada.
- `ffmpeg-static` 5.3.0 declara `GPL-3.0-or-later` no metadata npm local. A distribuição do executável e a composição/configuração do binário ainda precisam de uma revisão específica.
- O metadata local de `ultralytics` 8.4.131 declara `AGPL-3.0`. Em outubro ele foi separado em `requirements-experiments.txt`, mas a revisão de novembro confirmou que o aplicativo V4 também o usa pela mineração carregada dinamicamente. A receita CPU inclui essa dependência.
- Os identificadores de licença dos outros pacotes diretos Python e npm foram observados nos metadados instalados. Isso não cobre a árvore transitiva completa nem substitui a conferência dos textos e avisos exigidos para distribuição.

## Dependências diretas com atenção específica

| Componente | Versão | Identificador observado | Observação |
|---|---:|---|---|
| Ultralytics | 8.4.131 | AGPL-3.0 | Necessário à mineração carregada pelo aplicativo V4; incluído no build CPU local. |
| ffmpeg-static | 5.3.0 | GPL-3.0-or-later | Dependência declarada do desktop; auditar também o executável incluído e avisos de terceiros. |
| PyTorch | 2.9.1+rocm7.2.1 | BSD-3-Clause | Metadata local do pacote. |
| OpenCV Python | 5.0.0.93 | Apache-2.0 | Metadata local do pacote. |
| React / React DOM | 19.3.0 | MIT | Metadata local dos pacotes npm. |
| Electron / Vite | 44.9.3 / 8.3.5 | MIT | Metadata local dos pacotes npm. |
| 7zr standalone | 26.06 | LGPL-2.1-or-later | Extrator incluído no launcher offline; License.txt, copying.txt e referência ao código-fonte são gravados na preparação. Fonte: https://github.com/ip7z/7zip/tree/26.06. |

A [página oficial de licenciamento da Ultralytics](https://www.ultralytics.com/license) descreve AGPL-3.0 como licença padrão para YOLO e apresenta uma licença Enterprise para determinados usos sem as obrigações AGPL. Isso não determina, por si só, quais obrigações se aplicam ao projeto ou à forma de entrega acadêmica; a instituição/autoria deve confirmar antes de publicar ou distribuir os componentes correspondentes.

## Decisão para o rascunho de entrega

- Manter o pacote como `DRAFT_NOT_A_RELEASE`.
- Não adicionar uma licença ao código do projeto sem decisão expressa dos titulares/autoria.
- Não incluir vídeos, pesos, checkpoints, dados de terceiros ou derivados TEST até documentar a origem e a autorização aplicável.
- A distribuição do aplicativo V4 precisa considerar Ultralytics como dependência efetiva; os testes de pacote local não encerram sua revisão de licenciamento.
- Tratar FFmpeg como bloqueio de distribuição da aplicação autocontida até revisar o binário e os avisos/licenças de terceiros.
- Gerar uma lista completa de licenças transitivas e os avisos correspondentes apenas depois de definir o artefato efetivo que será distribuído.
