# Auditoria preliminar de licenças e direitos do pacote

**Data:** 26/10/2026<br>
**Estado:** inventário técnico preliminar; não é parecer jurídico nem autorização de distribuição.

## Achados no checkout

- Não foram encontrados `LICENSE`, `COPYING` ou `NOTICE` na raiz ou entre os arquivos rastreados do projeto. O repositório, portanto, não declara aqui uma licença para o código autoral do TCC.
- Os vídeos, clipes, anotações-fonte, pesos/checkpoints e dados externos ficam excluídos do pacote por padrão. A origem, licença e autorização de redistribuição não estão comprovadas de forma suficiente no checkout.
- O catálogo de timestamps TEST e os CSVs de candidatos/resultados são derivados desses dados. Mesmo sem vídeo, a permissão para redistribuir tais derivados precisa ser confirmada.
- `ffmpeg-static` 5.3.0 declara `GPL-3.0-or-later` no metadata npm local. A distribuição do executável e a composição/configuração do binário ainda precisam de uma revisão específica.
- O metadata local de `ultralytics` 8.4.131 declara `AGPL-3.0`. O pacote foi removido de `requirements.txt` e movido para `requirements-experiments.txt`, pois o aplicativo V4 não importa Ultralytics. Os scripts experimentais que o usam continuam sujeitos à revisão própria.
- Os identificadores de licença dos outros pacotes diretos Python e npm foram observados nos metadados instalados. Isso não cobre a árvore transitiva completa nem substitui a conferência dos textos e avisos exigidos para distribuição.

## Dependências diretas com atenção específica

| Componente | Versão | Identificador observado | Observação |
|---|---:|---|---|
| Ultralytics | 8.4.131 | AGPL-3.0 | Opcional; usado em scripts experimentais antigos, não importado pelo aplicativo V4. |
| ffmpeg-static | 5.3.0 | GPL-3.0-or-later | Dependência declarada do desktop; auditar também o executável incluído e avisos de terceiros. |
| PyTorch | 2.9.1+rocm7.2.1 | BSD-3-Clause | Metadata local do pacote. |
| OpenCV Python | 5.0.0.93 | Apache-2.0 | Metadata local do pacote. |
| React / React DOM | 19.3.0 | MIT | Metadata local dos pacotes npm. |
| Electron / Vite | 44.9.3 / 8.3.5 | MIT | Metadata local dos pacotes npm. |

A [página oficial de licenciamento da Ultralytics](https://www.ultralytics.com/license) descreve AGPL-3.0 como licença padrão para YOLO e apresenta uma licença Enterprise para determinados usos sem as obrigações AGPL. Isso não determina, por si só, quais obrigações se aplicam ao projeto ou à forma de entrega acadêmica; a instituição/autoria deve confirmar antes de publicar ou distribuir os componentes correspondentes.

## Decisão para o rascunho de entrega

- Manter o pacote como `DRAFT_NOT_A_RELEASE`.
- Não adicionar uma licença ao código do projeto sem decisão expressa dos titulares/autoria.
- Não incluir vídeos, pesos, checkpoints, dados de terceiros ou derivados TEST até documentar a origem e a autorização aplicável.
- Manter Ultralytics no grupo experimental opcional; omitir scripts/artefatos dependentes dele de qualquer subconjunto distribuído até concluir a revisão.
- Tratar FFmpeg como bloqueio de distribuição da aplicação autocontida até revisar o binário e os avisos/licenças de terceiros.
- Gerar uma lista completa de licenças transitivas e os avisos correspondentes apenas depois de definir o artefato efetivo que será distribuído.
