const { execFile } = require('node:child_process');

const MODES = { cpu: 'CPU', nvidia: 'GPU NVIDIA', amd: 'GPU AMD' };

function adapterVendor(adapter) {
  const id = String(adapter.PNPDeviceID || '').toUpperCase();
  if (id.includes('VEN_10DE')) return 'nvidia';
  if (id.includes('VEN_1002')) return 'amd';
  if (id.includes('VEN_8086')) return 'intel';
  const name = String(adapter.Name || '');
  if (/NVIDIA/i.test(name)) return 'nvidia';
  if (/AMD|Radeon/i.test(name)) return 'amd';
  if (/Intel/i.test(name)) return 'intel';
  return 'unknown';
}

function hardwareError(mode, hardware) {
  if (!MODES[mode]) return { code: 'INVALID_MODE', error: 'Escolha CPU, GPU NVIDIA ou GPU AMD.' };
  if (mode === 'cpu' || !hardware.known) return null;
  if (hardware.adapters.some((adapter) => adapterVendor(adapter) === mode)) return null;
  const names = hardware.adapters.map((adapter) => adapter.Name).filter(Boolean).join(', ');
  return {
    code: 'GPU_HARDWARE_NOT_FOUND',
    error: `Nenhuma ${MODES[mode]} foi encontrada neste computador.${names ? ` Placas detectadas: ${names}.` : ''} Escolha CPU ou outra GPU disponível.`,
  };
}

function detectHardware() {
  if (process.platform !== 'win32') return Promise.resolve({ known: false, adapters: [] });
  const script = '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Select-Object Name,PNPDeviceID,DriverVersion) | ConvertTo-Json -Compress';
  return new Promise((resolve) => {
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script],
      { windowsHide: true, timeout: 20000, encoding: 'utf8', maxBuffer: 256 * 1024 }, (error, stdout) => {
        if (error) return resolve({ known: false, adapters: [], error: 'Não foi possível consultar as placas pelo Windows.' });
        try {
          const decoded = JSON.parse(stdout.replace(/^\uFEFF/, '').trim() || '[]');
          resolve({ known: true, adapters: Array.isArray(decoded) ? decoded : [decoded] });
        } catch { resolve({ known: false, adapters: [], error: 'A consulta de placas não retornou dados válidos.' }); }
      });
  });
}

module.exports = { MODES, adapterVendor, hardwareError, detectHardware };
