const test = require('node:test');
const assert = require('node:assert/strict');
const { adapterVendor, hardwareError } = require('./accelerators.cjs');
const { backendCommand } = require('./backend-command.cjs');

test('NVIDIA é recusada em uma máquina apenas Intel', () => {
  const hardware = { known: true, adapters: [{ Name: 'Intel Iris Xe', PNPDeviceID: 'PCI\\VEN_8086' }] };
  assert.equal(hardwareError('nvidia', hardware).code, 'GPU_HARDWARE_NOT_FOUND');
  assert.match(hardwareError('nvidia', hardware).error, /Nenhuma GPU NVIDIA/);
  assert.equal(hardwareError('amd', hardware).code, 'GPU_HARDWARE_NOT_FOUND');
  assert.equal(hardwareError('cpu', hardware), null);
});

test('vendor PCI continua identificável com driver genérico', () => {
  assert.equal(adapterVendor({ Name: 'Microsoft Basic Display', PNPDeviceID: 'PCI\\VEN_1002&DEV_0000' }), 'amd');
  assert.equal(adapterVendor({ Name: 'NVIDIA RTX' }), 'nvidia');
});

test('falha da consulta Windows exige probe em vez de afirmar ausência', () => {
  assert.equal(hardwareError('nvidia', { known: false, adapters: [] }), null);
});

test('GPU instalada sem componente correspondente gera erro explícito', () => {
  assert.throws(() => backendCommand({ packaged: true, resourcesPath: '/app', userData: '/user', mode: 'nvidia', exists: () => false }),
    (error) => error.code === 'RUNTIME_NOT_INSTALLED');
});

test('NVIDIA usa seu componente e preserva o índice validado', () => {
  const command = backendCommand({ packaged: true, resourcesPath: '/app', userData: '/user', mode: 'nvidia', device: 'cuda:1', exists: () => true });
  assert.match(command.executable, /nvidia/);
  assert.equal(command.env.GOAL_REVIEW_ACCELERATOR, 'nvidia');
  assert.equal(command.env.GOAL_REVIEW_DEVICE, 'cuda:1');
});
