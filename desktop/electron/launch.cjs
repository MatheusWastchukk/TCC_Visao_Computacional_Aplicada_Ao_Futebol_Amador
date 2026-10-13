const { spawn } = require('child_process');
const electronPath = require('electron');

function launchElectron(args, extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  delete env.ELECTRON_RUN_AS_NODE;
  if (!Object.hasOwn(extraEnv, 'VITE_DEV_SERVER_URL')) delete env.VITE_DEV_SERVER_URL;
  return spawn(electronPath, args, {
    cwd: process.cwd(),
    env,
    stdio: 'inherit',
    windowsHide: false,
  });
}

module.exports = launchElectron;

if (require.main === module) {
  const child = launchElectron(process.argv.slice(2));
  child.on('error', (error) => {
    console.error(`[Electron] Não foi possível iniciar: ${error.message}`);
    process.exitCode = 1;
  });
  child.on('exit', (code, signal) => {
    process.exitCode = code ?? (signal ? 1 : 0);
  });
}
