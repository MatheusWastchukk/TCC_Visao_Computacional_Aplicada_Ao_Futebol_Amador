const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

let api;
const activeDownloads = new Map();

function apiIsReady() {
  return new Promise((resolve) => {
    const request = http.get('http://127.0.0.1:8765/health', (response) => {
      let body = '';
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        try { resolve(response.statusCode === 200 && JSON.parse(body).version === '0.4.0'); }
        catch { resolve(false); }
      });
    });
    request.on('error', () => resolve(false));
    request.setTimeout(500, () => { request.destroy(); resolve(false); });
  });
}

async function waitForApi() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (await apiIsReady()) return true;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
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
}

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
  Menu.setApplicationMenu(null);
  const root = path.join(__dirname, '..', '..');
  api = spawn(path.join(root, '.venv-rocm', 'Scripts', 'python.exe'), ['-m', 'uvicorn', 'backend.server:app', '--port', '8765'], { cwd: root, windowsHide: true });
  api.stdout.on('data', (data) => process.stdout.write(`[API] ${data}`));
  api.stderr.on('data', (data) => process.stderr.write(`[API] ${data}`));
  if (!await waitForApi()) {
    dialog.showErrorBox('Não foi possível iniciar a API', 'A interface será aberta, mas a análise de vídeo não estará disponível. Feche as janelas do aplicativo e tente novamente.');
  }
  createWindow();
});

app.on('will-quit', () => { if (api) api.kill(); });
