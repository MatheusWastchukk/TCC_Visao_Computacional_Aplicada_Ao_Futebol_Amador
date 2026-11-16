const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { probeDevice } = require('./device-probe.cjs');

test('erro 448 no início do backend é explicado como bloqueio de caminho', async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ger-probe-test-'));
  t.after(() => {
    assert.equal(path.dirname(root), os.tmpdir());
    assert(path.basename(root).startsWith('ger-probe-test-'));
    fs.rmSync(root, { recursive: true });
  });
  const command = { executable: process.execPath,
    args: ['-e', 'console.error("OSError: [WinError 448] ponto de montagem não confiável"); process.exit(2)', '--'],
    cwd: root, env: { ...process.env, MIOPEN_USER_DB_PATH: path.join(root, 'db'), MIOPEN_CUSTOM_CACHE_DIR: path.join(root, 'kernels') } };
  const result = await probeDevice(command, 'amd', root, 10000);
  assert.equal(result.code, 'RUNTIME_PATH_BLOCKED');
  assert.match(result.error, /448/);
  assert.doesNotMatch(result.error, /driver/i);
  assert.equal(path.dirname(result.diagnostic_log), path.join(root, 'diagnostics'));
});
