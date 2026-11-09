const { spawn, execFile } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const { version } = require('../package.json');
const { MODES } = require('./accelerators.cjs');

function stopProcess(child) {
  if (!child || child.exitCode !== null) return;
  if (process.platform === 'win32') {
    execFile('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {});
  } else child.kill();
}

function probeDevice(command, mode, userData, timeoutMs = 180000, signal) {
  const folder = path.join(userData, 'diagnostics');
  fs.mkdirSync(folder, { recursive: true });
  for (const key of ['MIOPEN_USER_DB_PATH', 'MIOPEN_CUSTOM_CACHE_DIR']) {
    fs.mkdirSync(command.env[key], { recursive: true });
  }
  const id = randomUUID();
  const output = path.join(folder, `device-${id}.json`);
  const log = fs.createWriteStream(path.join(folder, `device-${id}.log`));
  return new Promise((resolve) => {
    const child = spawn(command.executable, [...command.args, '--probe-device', mode, '--probe-output', output],
      { cwd: command.cwd, env: command.env, windowsHide: true });
    let settled = false;
    function finish(result) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      log.end();
      resolve(result);
    }
    const timer = setTimeout(() => {
      stopProcess(child);
      finish({ ok: false, code: 'PROBE_TIMEOUT', error: 'A verificação demorou mais que o esperado. Verifique o driver ou escolha CPU.' });
    }, timeoutMs);
    function onAbort() {
      stopProcess(child);
      finish({ ok: false, code: 'PROBE_CANCELLED', error: 'Verificação cancelada.' });
    }
    signal?.addEventListener('abort', onAbort, { once: true });
    if (signal?.aborted) onAbort();
    child.stdout?.on('data', (chunk) => log.write(chunk));
    child.stderr?.on('data', (chunk) => log.write(chunk));
    child.on('error', (error) => finish({ ok: false, code: 'RUNTIME_START_FAILED', error: `Não foi possível iniciar o componente de processamento: ${error.message}` }));
    child.on('close', (code) => {
      try {
        const report = JSON.parse(fs.readFileSync(output, 'utf8'));
        if (report.version !== version || report.mode !== mode) {
          finish({ ok: false, code: 'RUNTIME_VERSION_MISMATCH', error: 'O componente de processamento pertence a outra versão do aplicativo. Atualize o componente ou escolha CPU.' });
        } else if (code === 0 && report.ok && (mode === 'cpu' ? report.device === 'cpu' : /^cuda:\d+$/.test(report.device))) {
          finish(report);
        } else finish({ ...report, ok: false, error: report.code === 'VALIDATION_FAILED'
          ? `Não foi possível validar ${MODES[mode]}. Verifique o componente e o driver. Detalhe: ${report.error}`
          : report.error || 'A configuração escolhida não passou na validação.' });
      } catch {
        finish({ ok: false, code: 'PROBE_FAILED', error: 'O componente não concluiu a verificação. Verifique a instalação e o driver ou escolha CPU.' });
      }
    });
  });
}

module.exports = { probeDevice, stopProcess };
