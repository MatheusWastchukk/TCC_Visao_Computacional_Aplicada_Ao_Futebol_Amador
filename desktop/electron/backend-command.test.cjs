const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { version } = require('../package.json');
const { backendCommand } = require('./backend-command.cjs');

test('pacote usa backend incluído e CPU, sem Python externo', () => {
  const command = backendCommand({ packaged: true, resourcesPath: '/resources', userData: '/user', env: { GOAL_REVIEW_DEVICE: 'cuda:0' } });
  assert.equal(command.executable, path.join('/resources', 'backend', 'goal-review-api.exe'));
  assert.deepEqual(command.args, []);
  assert.equal(command.env.GOAL_REVIEW_DEVICE, 'cpu');
  assert.equal(command.env.GOAL_REVIEW_DATA_DIR, path.join('/user', 'data'));
});

test('desenvolvimento permite escolher Python e dispositivo explicitamente', () => {
  const command = backendCommand({ packaged: false, root: '/repo', env: { GOAL_REVIEW_PYTHON: '/python', GOAL_REVIEW_DEVICE: 'cpu' } });
  assert.equal(command.executable, '/python');
  assert.deepEqual(command.args, ['-m', 'backend.launcher']);
  assert.equal(command.cwd, '/repo');
});

test('GPU baixada usa o backend compartilhado; CPU permanece independente', () => {
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'ger-command-'));
  try {
    const mode = path.join(folder, 'gpu', 'nvidia'); fs.mkdirSync(mode, { recursive: true });
    fs.writeFileSync(path.join(mode, 'runtime.json'), JSON.stringify({ schema: 1, mode: 'nvidia', app_version: version, folder: 'payload-0123456789abcdef' }));
    const gpu = backendCommand({ packaged: true, resourcesPath: folder, userData: folder, mode: 'nvidia' });
    assert.equal(gpu.executable, path.join(folder, 'backend', 'goal-review-api.exe'));
    assert.match(gpu.env.GOAL_REVIEW_VENDOR_ROOT, /payload-0123456789abcdef$/);
    const cpu = backendCommand({ packaged: true, resourcesPath: folder, userData: folder, mode: 'cpu', env: gpu.env });
    assert.equal(cpu.env.GOAL_REVIEW_VENDOR_ROOT, undefined);
    const bad = { schema: 1, mode: 'nvidia', app_version: version, folder: '../outside' };
    fs.writeFileSync(path.join(mode, 'runtime.json'), JSON.stringify(bad));
    assert.throws(() => backendCommand({ packaged: true, resourcesPath: folder, userData: folder, mode: 'nvidia' }), /reinstalado/);
  } finally {
    // O alvo é criado por este teste, dentro do diretório temporário do Windows.
    assert.equal(path.dirname(folder), os.tmpdir());
    assert(path.basename(folder).startsWith('ger-command-'));
    fs.rmSync(folder, { recursive: true });
  }
});
