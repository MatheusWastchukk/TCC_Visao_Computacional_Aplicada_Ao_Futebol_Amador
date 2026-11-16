const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const test = require('node:test');
const { runtimePath } = require('./runtime-path.cjs');

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ger-junction-test-'));
  t.after(() => {
    assert.equal(path.dirname(root), os.tmpdir());
    assert(path.basename(root).startsWith('ger-junction-test-'));
    fs.rmSync(root, { recursive: true });
  });
  const target = path.join(root, 'vendor'); fs.mkdirSync(target);
  const executable = path.join(target, 'runtime.marker'); fs.writeFileSync(executable, 'preservar');
  const id = createHash('sha256').update(target.toLowerCase()).digest('hex').slice(0, 12);
  const alias = path.join(root, 'ger', id);
  return { root, target, executable, alias };
}

test('AMD cria e reutiliza a junção sem stat/realpath JavaScript no link', { skip: process.platform !== 'win32' }, (t) => {
  const { root, executable, alias } = fixture(t);
  let nativeCalls = 0;
  const legacy = () => { throw Object.assign(new Error('UNKNOWN: unknown error, stat'), { code: 'UNKNOWN' }); };
  legacy.native = (value) => {
    nativeCalls++;
    assert.notEqual(value, alias);
    return fs.realpathSync.native(value);
  };
  const filesystem = { ...fs, realpathSync: legacy, statSync: legacy };
  const first = runtimePath(executable, 'amd', { filesystem, temporary: root });
  assert.equal(path.dirname(first), alias);
  assert(nativeCalls > 0);
  assert.equal(fs.readFileSync(first, 'utf8'), 'preservar');
  assert.equal(runtimePath(executable, 'amd', { filesystem, temporary: root }), first);
});

test('uma pasta comum ocupando o alias é preservada', { skip: process.platform !== 'win32' }, (t) => {
  const { root, executable, alias } = fixture(t);
  fs.mkdirSync(alias, { recursive: true }); fs.writeFileSync(path.join(alias, 'user.txt'), 'preservar');
  assert.throws(() => runtimePath(executable, 'amd', { temporary: root }), { code: 'AMD_RUNTIME_PATH_OCCUPIED' });
  assert.equal(fs.readFileSync(path.join(alias, 'user.txt'), 'utf8'), 'preservar');
});

test('junção com outro destino é recusada sem remover o destino', { skip: process.platform !== 'win32' }, (t) => {
  const { root, executable, alias } = fixture(t);
  const other = path.join(root, 'other'); fs.mkdirSync(other); fs.writeFileSync(path.join(other, 'user.txt'), 'preservar');
  fs.mkdirSync(path.dirname(alias), { recursive: true }); fs.symlinkSync(other, alias, 'junction');
  assert.throws(() => runtimePath(executable, 'amd', { temporary: root }), { code: 'AMD_RUNTIME_PATH_OCCUPIED' });
  assert.equal(fs.readFileSync(path.join(other, 'user.txt'), 'utf8'), 'preservar');
});
