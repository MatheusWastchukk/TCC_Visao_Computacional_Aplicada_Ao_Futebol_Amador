// electron-builder 26: configuração do pacote preliminar Windows x64.
module.exports = {
  appId: 'br.edu.tcc.goaleventreview',
  productName: 'Goal Event Review',
  directories: { output: '../dist/cpu' },
  files: ['dist/**/*', 'electron/**/*', '!electron/**/*.test.cjs', 'package.json'],
  extraResources: [{ from: '../build/cpu/backend/goal-review-api', to: 'backend' }],
  asar: true,
  npmRebuild: false,
  win: { target: [{ target: 'nsis', arch: ['x64'] }], signAndEditExecutable: false },
  nsis: { oneClick: false, perMachine: false, allowToChangeInstallationDirectory: true },
  artifactName: '${productName}-${version}-cpu-${arch}.${ext}',
  publish: null,
};
