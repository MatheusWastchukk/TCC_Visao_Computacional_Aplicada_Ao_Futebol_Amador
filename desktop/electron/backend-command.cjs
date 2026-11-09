const fs = require('node:fs');
const path = require('node:path');

function backendCommand({ packaged, resourcesPath, root, userData, env = process.env, exists = fs.existsSync }) {
  const childEnv = { ...env, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1', GOAL_REVIEW_DEVICE: 'cpu' };
  if (packaged) {
    childEnv.GOAL_REVIEW_DATA_DIR = path.join(userData, 'data');
    return { executable: path.join(resourcesPath, 'backend', 'goal-review-api.exe'), args: [], cwd: userData, env: childEnv };
  }
  const candidates = ['.venv-cpu', '.venv'].map(name => path.join(root, name, 'Scripts', 'python.exe'));
  const executable = env.GOAL_REVIEW_PYTHON || candidates.find(candidate => exists(candidate));
  if (!executable) throw new Error('Prepare o ambiente Python CPU antes de iniciar o aplicativo.');
  return { executable, args: ['-m', 'backend.launcher'], cwd: root, env: childEnv };
}

module.exports = { backendCommand };
