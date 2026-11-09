const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');

function runtimePath(executable, mode) {
  if (process.platform !== 'win32' || mode !== 'amd') return executable;
  // As bibliotecas nativas ROCm ainda falham com caminhos longos. A junção
  // mantém o runtime incluído no pacote e oferece um caminho curto ao processo.
  const target = path.dirname(executable);
  const id = createHash('sha256').update(target.toLowerCase()).digest('hex').slice(0, 12);
  const folder = path.join(os.tmpdir(), 'ger');
  const alias = path.join(folder, id);
  fs.mkdirSync(folder, { recursive: true });
  try { fs.symlinkSync(target, alias, 'junction'); }
  catch (error) { if (error.code !== 'EEXIST') throw new Error(`Não foi possível preparar o caminho do componente AMD: ${error.message}`); }
  if (fs.realpathSync(alias).toLowerCase() !== fs.realpathSync(target).toLowerCase()) {
    throw new Error('O caminho temporário do componente AMD está ocupado. Limpe a pasta temporária do aplicativo e tente novamente.');
  }
  return path.join(alias, path.basename(executable));
}

module.exports = { runtimePath };
