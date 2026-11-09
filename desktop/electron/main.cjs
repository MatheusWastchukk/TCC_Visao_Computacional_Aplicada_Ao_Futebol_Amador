const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');
const { backendCommand } = require('./backend-command.cjs');
const { detectHardware, hardwareError } = require('./accelerators.cjs');
const { probeDevice, stopProcess } = require('./device-probe.cjs');
const { randomUUID } = require('node:crypto');
const fs = require('node:fs');
const { version } = require('../package.json');
const primaryInstance = app.requestSingleInstanceLock();
if (!primaryInstance) app.quit();

let api;
let modeWindow;
let appWindow;
let starting = false;
let probeController;
const activeDownloads = new Map();

function apiIsReady(instanceId) {
  return new Promise((resolve) => {
    const request = http.get('http://127.0.0.1:8765/health', (response) => {
      let body = '';
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        try {
          const health = JSON.parse(body);
          resolve(response.statusCode === 200 && health.version === version && health.instance_id === instanceId);
        }
        catch { resolve(false); }
      });
    });
    request.on('error', () => resolve(false));
    request.setTimeout(500, () => { request.destroy(); resolve(false); });
  });
}

async function waitForApi(child, instanceId, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await apiIsReady(instanceId)) return { ready: true };
    if (child.spawnError) return { ready: false, reason: `Falha ao iniciar o Python: ${child.spawnError.message}` };
    if (child.exitCode !== null) return { ready: false, reason: `O processo da API encerrou com código ${child.exitCode}.` };
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return { ready: false, reason: `A API não respondeu em ${Math.round(timeoutMs / 1000)} segundos.` };
}

function createWindow() {
  const win = new BrowserWindow({ width: 1440, height: 920, webPreferences: { preload: path.join(__dirname, 'preload.cjs') } });
  if (!app.isPackaged && process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
  win.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
    if (isMainFrame) console.error(`[Renderer] Falha ao carregar ${url}: ${code} ${description}`);
  });
  win.webContents.on('console-message', (_event, level, message, line, sourceId) => {
    if (level >= 2) console.error(`[Renderer] ${message} (${sourceId}:${line})`);
  });
  win.webContents.on('render-process-gone', (_event, details) => {
    console.error(`[Renderer] Processo encerrado: ${details.reason} (código ${details.exitCode})`);
  });
  win.webContents.on('preload-error', (_event, preloadPath, error) => {
    console.error(`[Preload] Falha em ${preloadPath}: ${error.message}`);
  });
  return win;
}

ipcMain.handle('processing-start', async (event, mode) => {
  if (!modeWindow || event.sender !== modeWindow.webContents || starting) {
    return { ok: false, error: 'Aguarde a verificação em andamento.' };
  }
  starting = true;
  try {
    if (mode !== 'cpu') {
      const error = hardwareError(mode, await detectHardware());
      if (error) return { ok: false, ...error };
    }
    const root = path.join(__dirname, '..', '..');
    const userData = app.getPath('userData');
    const options = { packaged: app.isPackaged, resourcesPath: process.resourcesPath, root, userData, mode };
    let command = backendCommand(options);
    probeController = new AbortController();
    const report = await probeDevice(command, mode, userData, 180000, probeController.signal);
    probeController = null;
    if (!report.ok) return report;
    if (!modeWindow || modeWindow.isDestroyed()) return { ok: false, error: 'Inicialização cancelada.' };
    command = backendCommand({ ...options, device: report.device });
    command.env.GOAL_REVIEW_INSTANCE_ID = randomUUID();
    const folder = path.join(userData, 'diagnostics');
    const log = fs.createWriteStream(path.join(folder, `api-${command.env.GOAL_REVIEW_INSTANCE_ID}.log`));
    let stderr = '';
    api = spawn(command.executable, command.args, { cwd: command.cwd, env: command.env, windowsHide: true });
    api.on('error', (error) => { api.spawnError = error; });
    api.stdout.on('data', (chunk) => { log.write(chunk); process.stdout.write(`[API] ${chunk}`); });
    api.stderr.on('data', (chunk) => { stderr = (stderr + chunk.toString()).slice(-4000); log.write(chunk); process.stderr.write(`[API] ${chunk}`); });
    api.on('close', () => log.end());
    const ready = await waitForApi(api, command.env.GOAL_REVIEW_INSTANCE_ID);
    if (!ready.ready) {
      stopProcess(api);
      api = null;
      const busy = /address already in use|10048|eaddrinuse/i.test(stderr);
      return { ok: false, code: 'API_START_FAILED', error: busy
        ? 'Outra instância está usando a porta do aplicativo. Feche essa instância e tente novamente.'
        : `Não foi possível iniciar o aplicativo com a opção selecionada. ${ready.reason}` };
    }
    appWindow = createWindow();
    modeWindow.close();
    return { ok: true };
  } catch (error) {
    probeController?.abort();
    probeController = null;
    stopProcess(api);
    api = null;
    return { ok: false, code: error.code || 'START_FAILED', error: error.message };
  } finally { starting = false; }
});

ipcMain.handle('select-video', async () => (await dialog.showOpenDialog({
  properties: ['openFile'], filters: [{ name: 'Vídeos', extensions: ['mp4', 'mov', 'avi', 'mkv'] }],
})).filePaths[0] || '');
ipcMain.handle('download-url', (event, { url, downloadId }) => new Promise((resolve, reject) => {
  const sender = event.sender;
  const timeout = setTimeout(() => reject(new Error('A preparação do download demorou mais que o esperado.')), 300000);
  const onWillDownload = (_downloadEvent, item) => {
    clearTimeout(timeout);
    activeDownloads.set(downloadId, item);
    const report = () => sender.send('download-progress', {
      downloadId,
      receivedBytes: item.getReceivedBytes(),
      totalBytes: item.getTotalBytes(),
      state: item.getState(),
    });
    item.on('updated', report);
    report();
    item.once('done', (_doneEvent, state) => {
      activeDownloads.delete(downloadId);
      if (state === 'completed') resolve({ filename: item.getFilename() });
      else reject(new Error(state === 'cancelled' ? 'Download cancelado.' : 'O download não foi concluído.'));
    });
  };
  sender.session.once('will-download', onWillDownload);
  sender.downloadURL(url);
}));
ipcMain.handle('cancel-download', (_event, downloadId) => {
  activeDownloads.get(downloadId)?.cancel();
  return { ok: true };
});

// A interface deve abrir mesmo em máquinas ou ambientes sem um processo gráfico utilizável.
app.disableHardwareAcceleration();
app.commandLine.appendSwitch('disable-gpu');

app.whenReady().then(async () => {
  if (!primaryInstance) return;
  Menu.setApplicationMenu(null);
  modeWindow = new BrowserWindow({ width: 880, height: 620, minWidth: 640, minHeight: 540,
    webPreferences: { preload: path.join(__dirname, 'mode-preload.cjs') } });
  modeWindow.loadFile(path.join(__dirname, 'mode-selector.html'));
  modeWindow.on('closed', () => { modeWindow = null; if (!appWindow) probeController?.abort(); });
});

app.on('before-quit', () => { probeController?.abort(); stopProcess(api); });
app.on('second-instance', () => {
  const window = appWindow && !appWindow.isDestroyed() ? appWindow : modeWindow;
  if (window) { if (window.isMinimized()) window.restore(); window.focus(); }
});
