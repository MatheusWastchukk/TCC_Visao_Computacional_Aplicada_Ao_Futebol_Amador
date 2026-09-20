const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

let api;

function apiIsReady() {
  return new Promise((resolve) => {
    const request = http.get('http://127.0.0.1:8765/health', (response) => {
      let body = '';
      response.on('data', (chunk) => { body += chunk; });
      response.on('end', () => {
        try { resolve(response.statusCode === 200 && JSON.parse(body).version === '0.3.7'); }
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
  win.loadURL('http://localhost:5173');
}

ipcMain.handle('select-video', async () => (await dialog.showOpenDialog({
  properties: ['openFile'], filters: [{ name: 'Vídeos', extensions: ['mp4', 'mov', 'avi', 'mkv'] }],
})).filePaths[0] || '');
ipcMain.handle('download-url', (event, url) => new Promise((resolve, reject) => {
  const timeout = setTimeout(() => reject(new Error('A preparação do download demorou mais que o esperado.')), 300000);
  event.sender.session.once('will-download', (_downloadEvent, item) => {
    clearTimeout(timeout);
    resolve({ filename: item.getFilename() });
  });
  event.sender.downloadURL(url);
}));

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
