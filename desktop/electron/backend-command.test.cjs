const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
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
