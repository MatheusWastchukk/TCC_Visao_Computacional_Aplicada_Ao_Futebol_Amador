const fs = require('node:fs');
const path = require('node:path');
const base = require('./electron-builder.cjs');
const { version } = require('./package.json');

const gpuResources = ['nvidia', 'amd'].map((mode) => {
  const folder = `../build/engines/${mode}/goal-review-api`;
  if (!fs.existsSync(path.join(__dirname, folder, 'goal-review-api.exe'))) {
    throw new Error(`Construa primeiro o componente ${mode} com scripts/build_accelerator.ps1.`);
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, folder, 'component.json'), 'utf8').replace(/^\uFEFF/, ''));
  if (manifest.mode !== mode || manifest.app_version !== version) throw new Error(`O componente ${mode} precisa pertencer à versão ${version}.`);
  return { from: folder, to: `engines/${mode}` };
});
const cpu = JSON.parse(fs.readFileSync(path.join(__dirname, '../build/cpu/backend/goal-review-api/component.json'), 'utf8').replace(/^\uFEFF/, ''));
if (cpu.mode !== 'cpu' || cpu.app_version !== version) throw new Error(`Construa o componente CPU da versão ${version}.`);

module.exports = {
  ...base,
  directories: { output: '../dist/complete' },
  extraResources: [...base.extraResources, ...gpuResources],
  win: { ...base.win, target: [{ target: 'dir', arch: ['x64'] }] },
};
