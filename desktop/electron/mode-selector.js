const buttons = [...document.querySelectorAll('[data-mode]')];
const status = document.getElementById('status');
const error = document.getElementById('error');

for (const button of buttons) {
  button.addEventListener('click', async () => {
    for (const item of buttons) item.disabled = true;
    error.textContent = '';
    status.className = 'busy';
    status.textContent = `Verificando ${button.querySelector('b').textContent}… Aguarde.`;
    try {
      const result = await window.processing.start(button.dataset.mode);
      if (!result.ok) error.textContent = result.error;
      status.textContent = result.ok ? 'Abrindo o aplicativo…' : 'Escolha outra opção ou tente novamente.';
    } catch {
      error.textContent = 'Não foi possível verificar esta configuração. Tente novamente.';
      status.textContent = '';
    } finally {
      status.className = '';
      for (const item of buttons) item.disabled = false;
    }
  });
}
