const launchElectron = require('./launch.cjs');

async function main() {
  const { createServer } = await import('vite');
  const server = await createServer();
  await server.listen();
  const address = server.httpServer.address();
  const port = typeof address === 'object' && address ? address.port : 5173;
  const url = `http://localhost:${port}`;
  console.log(`[Vite] Servidor de desenvolvimento em ${url}`);

  const electron = launchElectron(['.'], { VITE_DEV_SERVER_URL: url });
  electron.on('error', async (error) => {
    console.error(`[Electron] Não foi possível iniciar: ${error.message}`);
    await server.close();
    process.exitCode = 1;
  });
  electron.on('exit', async (code, signal) => {
    await server.close();
    process.exitCode = code ?? (signal ? 1 : 0);
  });
}

main().catch((error) => {
  console.error(`[Desenvolvimento] Falha ao iniciar: ${error.message}`);
  process.exitCode = 1;
});
