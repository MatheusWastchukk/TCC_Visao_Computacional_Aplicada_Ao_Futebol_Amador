const fs = require('fs');
const path = require('path');
const { runtimePath } = require('./runtime-path.cjs');

function backendCommand({ packaged, resourcesPath, root, userData, mode = 'cpu', device, env = process.env, exists = fs.existsSync }) {
  if (!['cpu', 'nvidia', 'amd'].includes(mode)) throw new Error('Opção de processamento inválida.');
  const childEnv = { ...env, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1' };
  childEnv.GOAL_REVIEW_ACCELERATOR = mode;
  childEnv.GOAL_REVIEW_DEVICE = device || (mode === 'cpu' ? 'cpu' : 'cuda:0');
  if (packaged) childEnv.GOAL_REVIEW_DATA_DIR = path.join(userData, 'data');
  const cache = path.join(userData || root, 'cache', mode);
  childEnv.MIOPEN_USER_DB_PATH = path.join(cache, 'miopen-db');
  childEnv.MIOPEN_CUSTOM_CACHE_DIR = path.join(cache, 'miopen-kernels');
  if (packaged) {
    const executable = mode === 'cpu'
      ? path.join(resourcesPath, 'backend', 'goal-review-api.exe')
      : [path.join(resourcesPath, 'engines', mode, 'goal-review-api.exe'), path.join(userData, 'engines', mode, 'goal-review-api.exe')]
        .find((candidate) => exists(candidate));
    if (!executable) {
      const error = new Error(`O componente de processamento ${mode === 'nvidia' ? 'NVIDIA' : 'AMD'} não está instalado neste aplicativo. Instale o componente correspondente ou escolha CPU.`);
      error.code = 'RUNTIME_NOT_INSTALLED';
      throw error;
    }
    return {
      executable: runtimePath(executable, mode),
      args: [],
      cwd: userData,
      env: childEnv,
    };
  }
  const environments = { cpu: ['.venv-cpu', '.venv'], nvidia: ['.venv-cuda'], amd: ['.venv-rocm'] };
  const candidates = environments[mode].map((name) => path.join(root, name, 'Scripts', 'python.exe'));
  const explicit = mode === 'cpu' ? env.GOAL_REVIEW_PYTHON : env[`GOAL_REVIEW_${mode.toUpperCase()}_PYTHON`];
  const executable = explicit || candidates.find((candidate) => exists(candidate));
  if (!executable) {
    const error = new Error(`O ambiente de processamento ${mode} não foi preparado. Instale o componente correspondente ou escolha CPU.`);
    error.code = 'RUNTIME_NOT_INSTALLED';
    throw error;
  }
  return { executable, args: ['-m', 'backend.launcher'], cwd: root, env: childEnv };
}

module.exports = { backendCommand };
