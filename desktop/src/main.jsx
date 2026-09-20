import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Download, FolderOpen, Moon, Play, Sun, Video, Cpu, HardDrive } from 'lucide-react';
import './style.css';

const API = 'http://127.0.0.1:8765';
const TAGS = ['GOAL', 'NO_GOAL', 'NOT_EVALUABLE'];
const formatDuration = (seconds) => `${String(Math.floor(Number(seconds) / 60)).padStart(2, '0')}:${String(Math.round(Number(seconds) % 60)).padStart(2, '0')}`;
const clipNumber = (clip) => String(clip.segment_id.match(/(\d+)$/)?.[1] || clip.segment_id).padStart(3, '0');
const progressText = (status) => status?.status === 'completed'
  ? `Tempo total: ${formatDuration(status.elapsed)}`
  : (Number(status?.estimated_total || 0) - Number(status?.elapsed || 0) > 1 ? `Previsão restante: ${formatDuration(Number(status.estimated_total) - Number(status.elapsed))}` : 'Calculando previsão de término');

function App() {
  const [videoPath, setVideoPath] = useState(''); const [mode, setMode] = useState('alta_cobertura');
  const [sessionId, setSessionId] = useState(''); const [status, setStatus] = useState(null);
  const [clips, setClips] = useState([]); const [currentClip, setCurrentClip] = useState(null);
  const [selected, setSelected] = useState(new Set()); const [dark, setDark] = useState(false); const [error, setError] = useState(''); const [savingClips, setSavingClips] = useState(new Set()); const [preparingDownloads, setPreparingDownloads] = useState(new Set()); const [preparingCompilation, setPreparingCompilation] = useState(false);

  const startAnalysis = async () => {
    setError(''); setClips([]); setCurrentClip(null); setSelected(new Set()); setSessionId('');
    setStatus({ status: 'starting', stage: 'Iniciando análise', progress: 0 });
    try {
      const response = await fetch(`${API}/sessions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video_path: videoPath, mode }) });
      if (!response.ok) throw new Error();
      setSessionId((await response.json()).session_id);
    } catch { setStatus(null); setError('Não foi possível conectar à API local. Feche todas as janelas do aplicativo e inicie-o novamente.'); }
  };

  useEffect(() => {
    if (!sessionId) return undefined;
    const poll = setInterval(async () => {
      try {
        const response = await fetch(`${API}/sessions/${sessionId}`); if (!response.ok) throw new Error();
        const next = await response.json(); setStatus(next);
        if (next.status === 'completed') {
          const clipResponse = await fetch(`${API}/sessions/${sessionId}/clips`); const items = await clipResponse.json();
          setClips(items); setCurrentClip(items[0] || null); clearInterval(poll);
        }
        if (next.status === 'failed') { setError(next.error || 'A análise não foi concluída.'); clearInterval(poll); }
      } catch { setError('A conexão com a análise foi perdida.'); clearInterval(poll); }
    }, 500);
    return () => clearInterval(poll);
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !savingClips.size) return undefined;
    const pollRenders = setInterval(async () => {
      try {
        const response = await fetch(`${API}/sessions/${sessionId}/clips`);
        if (!response.ok) throw new Error();
        const items = await response.json();
        setClips(items);
        setCurrentClip((current) => items.find((item) => item.segment_id === current?.segment_id) || current);
        const pending = new Set(items.filter((item) => item.render_status === 'rendering').map((item) => item.segment_id));
        setSavingClips(pending);
        if (items.some((item) => item.render_status === 'failed')) setError('Não foi possível gerar um dos vídeos classificados. Consulte o terminal.');
      } catch { setError('Não foi possível acompanhar a geração do clipe.'); }
    }, 800);
    return () => clearInterval(pollRenders);
  }, [sessionId, savingClips]);

  const downloadUrl = (clip) => `${API}/sessions/${sessionId}/clips/${clip.segment_id}/download`;
  const downloadClip = async (clip) => {
    setPreparingDownloads((items) => new Set(items).add(clip.segment_id));
    try { await window.desktop.download(downloadUrl(clip)); }
    catch { setError('Não foi possível preparar o download do clipe. Consulte o terminal.'); }
    finally { setPreparingDownloads((items) => { const next = new Set(items); next.delete(clip.segment_id); return next; }); }
  };
  const downloadSelected = async () => {
    const query = [...selected].map((id) => `segment_ids=${encodeURIComponent(id)}`).join('&');
    setPreparingCompilation(true);
    try { await window.desktop.download(`${API}/sessions/${sessionId}/clips/compilation?${query}`); }
    catch { setError('Não foi possível preparar a compilação. Consulte o terminal.'); }
    finally { setPreparingCompilation(false); }
  };
  const toggleSelection = (id) => setSelected((previous) => { const next = new Set(previous); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const markClip = async (tag) => {
    if (!currentClip) return;
    const before = currentClip;
    const updated = { ...currentClip, human_label: tag };
    setSavingClips((items) => new Set(items).add(currentClip.segment_id)); setCurrentClip(updated); setClips((items) => items.map((item) => item.segment_id === updated.segment_id ? updated : item));
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/clips/${currentClip.segment_id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tag }) });
      if (!response.ok) throw new Error();
    } catch { setCurrentClip(before); setClips((items) => items.map((item) => item.segment_id === before.segment_id ? before : item)); setSavingClips((items) => { const next = new Set(items); next.delete(currentClip.segment_id); return next; }); setError('Não foi possível salvar a classificação do clipe.'); }
  };
  const metrics = status?.metrics;

  return <main className={dark ? 'dark' : ''}>
    <header><div><p>Análise de vídeo</p><h1>Goal Event Review</h1></div><button className="icon" onClick={() => setDark(!dark)} title="Alternar tema">{dark ? <Sun /> : <Moon />}</button></header>
    <section className="setup"><input readOnly value={videoPath} placeholder="Selecione um vídeo" /><button onClick={async () => setVideoPath(await window.desktop.selectVideo())}><FolderOpen />Selecionar vídeo</button><select value={mode} onChange={(event) => setMode(event.target.value)}><option value="alta_cobertura">Alta cobertura</option><option value="cobertura_reduzida">Cobertura reduzida</option><option value="triagem_rapida">Triagem rápida</option></select><button onClick={startAnalysis} disabled={!videoPath || status?.status === 'running'}><Play />Analisar</button></section>
    {error && <p className="notice error">{error}</p>}
    {status && <><section className="progress"><div className="progress-top"><div><b>{status.stage}</b><small>{progressText(status)}</small></div><strong>{Number(status.progress || 0).toFixed(2).replace('.', ',')}%</strong></div><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(Number(status.progress || 0), 100))}%` }} /></div><div className="progress-bottom"><span>0,00%</span><span>100,00%</span></div></section>{metrics && <section className="metrics"><div><Cpu /><span><b>{metrics.processor === 'GPU' ? 'Mineração pela GPU' : 'Mineração pela CPU'}</b><small>{metrics.device_name || metrics.cpu_name}</small></span></div><div><span><b>CPU do app {metrics.process_cpu_percent}%</b><small>Processo Python</small></span><span><b>RAM do app {metrics.process_ram_mb} MB</b><small>{metrics.process_ram_percent}% da memória</small></span></div>{metrics.processor === 'GPU' && <div><HardDrive /><span><b>VRAM do app {metrics.gpu_process_memory_mb} MB</b><small>GPU total: {metrics.gpu_memory_used_mb} / {metrics.gpu_memory_total_mb} MB</small></span></div>}</section>}</>}
    {!status && !clips.length && <section className="empty"><FolderOpen size={42} /><h2>Selecione um vídeo para começar</h2><p>Os clipes, o player e as opções de classificação aparecerão aqui após a análise.</p></section>}
    {status && !clips.length && status.status !== 'failed' && <section className="empty"><Video size={42} /><h2>{status.stage}</h2><p>A mineração está em andamento. Os clipes aparecerão nesta área quando estiverem prontos.</p></section>}
    {status?.status === 'failed' && <section className="empty"><Video size={42} /><h2>Não foi possível concluir a análise</h2><p>Corrija o erro informado acima e selecione o vídeo novamente.</p></section>}
    {clips.length > 0 && <section className="workspace"><aside><div className="listhead"><div><h2>Clipes</h2><small>{selected.size ? `${selected.size} selecionado(s)` : 'Selecione para compilar'}</small></div><button className="download-selected" disabled={preparingCompilation || !selected.size || clips.some((clip) => selected.has(clip.segment_id) && (savingClips.has(clip.segment_id) || clip.render_status === 'rendering'))} onClick={downloadSelected}><Download />{preparingCompilation ? 'Preparando…' : 'Baixar selecionados'}</button></div>{clips.map((clip) => <div className={`clip ${currentClip?.segment_id === clip.segment_id ? 'active' : ''}`} onClick={() => setCurrentClip(clip)} key={clip.segment_id}><input type="checkbox" checked={selected.has(clip.segment_id)} onClick={(event) => event.stopPropagation()} onChange={() => toggleSelection(clip.segment_id)} /><span className={`tag ${clip.human_label || clip.label || 'POSSIBLE_GOAL'}`}>{clip.human_label || clip.label || 'POSSIBLE_GOAL'}</span><b>Clipe {clipNumber(clip)}</b><small className={savingClips.has(clip.segment_id) || clip.render_status === 'rendering' || preparingDownloads.has(clip.segment_id) ? 'saving' : ''}>{formatDuration(clip.duration_seconds)}{(savingClips.has(clip.segment_id) || clip.render_status === 'rendering') && ' · Salvando…'}{preparingDownloads.has(clip.segment_id) && ' · Preparando download…'}</small></div>)}</aside>
      <article>{currentClip && <><div className="player"><video key={currentClip.clip_filename} controls src={`${API}/sessions/${sessionId}/clips/${currentClip.segment_id}/media`} /></div><div className="clip-details"><div><p>Resultado da análise</p><h2>Clipe {clipNumber(currentClip)}</h2><small>Duração: {formatDuration(currentClip.duration_seconds)}</small></div><button className="icon small" title="Baixar clipe" disabled={preparingDownloads.has(currentClip.segment_id) || savingClips.has(currentClip.segment_id) || currentClip.render_status === 'rendering'} onClick={() => downloadClip(currentClip)}><Download /></button></div><div className="tags"><div><b>Classificar clipe</b><small>{savingClips.has(currentClip.segment_id) ? 'Salvando classificação…' : 'A classificação será gravada no overlay e no nome do arquivo exportado.'}</small></div><div className="tag-buttons">{TAGS.map((tag) => <button disabled={savingClips.has(currentClip.segment_id)} className={`${tag} ${currentClip.human_label === tag ? 'selected-tag' : ''}`} onClick={() => markClip(tag)} key={tag}>{tag}</button>)}</div></div></>}</article></section>}
  </main>;
}
createRoot(document.getElementById('root')).render(<App />);
