// Os componentes ficam fora do ASAR; use as APIs originais no Electron.
const fs = require(process.versions.electron ? 'original-fs' : 'node:fs');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');

function runtimePath(executable, mode, { filesystem = fs, platform = process.platform, temporary = os.tmpdir() } = {}) {
  if (platform !== 'win32' || mode !== 'amd') return executable;
  // As bibliotecas nativas ROCm ainda falham com caminhos longos. A junção
  // mantém o runtime incluído no pacote e oferece um caminho curto ao processo.
  const target = path.dirname(executable);
  const id = createHash('sha256').update(target.toLowerCase()).digest('hex').slice(0, 12);
  const folder = path.join(temporary, 'ger');
  const alias = path.join(folder, id);
  const occupied = () => Object.assign(new Error('O caminho temporário do componente AMD está ocupado por outro arquivo ou pasta. Nenhum arquivo foi removido. Escolha CPU ou informe esse erro ao suporte.'), { code: 'AMD_RUNTIME_PATH_OCCUPIED' });
  try {
    filesystem.mkdirSync(folder, { recursive: true });
    try { filesystem.symlinkSync(target, alias, 'junction'); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    if (!filesystem.lstatSync(alias).isSymbolicLink()) throw occupied();
    // Não execute realpathSync/stat na junção: essa resolução em JavaScript
    // pode falhar com UNKNOWN no Windows. Leia o destino sem seguir o link
    // e compare os caminhos físicos pela API nativa.
    const linkedTarget = filesystem.readlinkSync(alias);
    const canonical = (value) => path.normalize(filesystem.realpathSync.native(value)).toLowerCase();
    if (canonical(linkedTarget) !== canonical(target)) throw occupied();
  } catch (error) {
    if (error.code === 'AMD_RUNTIME_PATH_OCCUPIED') throw error;
    const failure = new Error(`Não foi possível preparar o caminho temporário da GPU AMD. Os componentes instalados foram preservados. Detalhe: ${error.message}`);
    failure.code = 'AMD_RUNTIME_PATH_UNAVAILABLE';
    throw failure;
  }
  return path.join(alias, path.basename(executable));
}

module.exports = { runtimePath };
