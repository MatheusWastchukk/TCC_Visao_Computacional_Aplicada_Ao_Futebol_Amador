import React from 'react';

const percentage = (value) => value != null ? `${value}%` : '—';
const duration = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return '—';
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
};

export default function ProcessingDetails({ metrics, status }) {
  const gpuUsage = metrics?.processor !== 'GPU' ? 'Não utilizada'
    : metrics.gpu_utilization_percent != null ? percentage(metrics.gpu_utilization_percent)
      : metrics.gpu_utilization_status === 'warming_up' ? 'Aguardando leitura…' : 'Leitura indisponível';

  return <details className="technical-details processing-details">
    <summary>Detalhes do processamento</summary>
    <div className="processing-details-content">
      {metrics ? <section className="metrics" aria-label="Métricas do processamento">
        <div><span><b>Processador</b><small>{metrics.processor === 'GPU' ? metrics.device_name : metrics.cpu_name} · {metrics.processor}</small></span></div>
        <div><span><b title="Parcela da capacidade total de CPU usada pelo processo que analisa o vídeo. Mesmo no modo GPU, ele usa CPU para ler e preparar quadros.">CPU da análise</b><small>{percentage(metrics.process_cpu_percent)}</small></span></div>
        <div><span><b title="Memória usada pelo processo de análise, em relação à RAM total do computador.">Memória RAM</b><small>{metrics.process_ram_mb ?? '—'} MB · {percentage(metrics.process_ram_percent)}</small></span></div>
        <div><span><b title="Atividade atual da GPU selecionada, incluindo outros programas.">Uso de GPU</b><small title={metrics.gpu_utilization_error || undefined}>{gpuUsage}</small></span></div>
        <div><span><b title="Memória total em uso na GPU selecionada, incluindo outros programas.">Memória da GPU</b><small>{metrics.gpu_memory_used_mb != null ? `${metrics.gpu_memory_used_mb} / ${metrics.gpu_memory_total_mb} MB · ${percentage(metrics.gpu_memory_percent)}` : metrics.processor === 'GPU' ? 'Leitura indisponível' : 'Não utilizada'}</small></span></div>
        <div><span><b>Tempo de processamento</b><small>{duration(status?.elapsed)}</small></span></div>
      </section> : <p>Aguardando as métricas do processamento.</p>}
      <p className="metrics-explanation">CPU e RAM referem-se ao processo que faz a análise, sem incluir a interface nem os processos separados que geram os clipes. Mesmo usando GPU, a análise também utiliza CPU. O uso e a memória da GPU incluem outros programas.</p>
      <p className="processing-audit"><small>Aplicativo {status?.app_version || '—'} · Pipeline {status?.pipeline_version || '—'}<br />As impressões SHA-256 ficam nos arquivos de auditoria para identificar os modelos e a configuração usados.</small></p>
    </div>
  </details>;
}
