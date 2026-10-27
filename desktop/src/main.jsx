import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowLeft, ArrowRight, CheckCircle2, CircleAlert, CircleHelp, Download, FolderOpen, Moon, Play, Sun, Trash2, Video } from 'lucide-react';
import './style.css';

const API = 'http://127.0.0.1:8765';
const TAGS = ['GOAL', 'NO_GOAL', 'NOT_EVALUABLE'];
const formatDuration = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return '—';
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
};
const candidateNumber = (index) => String(index + 1).padStart(2, '0');
const reviewText = (status) => ({ completed: 'Concluída', deferred: 'Revisar depois', pending: 'Pendente' }[status] || 'Pendente');
const modeText = (mode) => ({ alta_cobertura: 'Alta cobertura', cobertura_reduzida: 'Cobertura reduzida', triagem_rapida: 'Triagem rápida' }[mode] || mode || '—');
const progressText = (status) => status?.status === 'completed'
  ? `Tempo total: ${formatDuration(status.elapsed)}`
  : (Number(status?.estimated_total || 0) - Number(status?.elapsed || 0) > 1 ? `Previsão restante: ${formatDuration(Number(status.estimated_total) - Number(status.elapsed))}` : 'Calculando previsão de término');

function App() {
  const [videoPath, setVideoPath] = useState('');
  const [mode, setMode] = useState('alta_cobertura');
  const [sessionId, setSessionId] = useState('');
  const [status, setStatus] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [clips, setClips] = useState([]);
  const [currentClip, setCurrentClip] = useState(null);
  const [selected, setSelected] = useState(new Set());
  const [dark, setDark] = useState(false);
  const [error, setError] = useState('');
  const [analysisErrorDialog, setAnalysisErrorDialog] = useState(null);
  const [notice, setNotice] = useState('');
  const [savingClips, setSavingClips] = useState(new Set());
  const [undoingDecision, setUndoingDecision] = useState(false);
  const [preparingDownloads, setPreparingDownloads] = useState(new Set());
  const [preparingCompilation, setPreparingCompilation] = useState(false);
  const [compilationProgress, setCompilationProgress] = useState(null);
  const [preparingExport, setPreparingExport] = useState('');
  const [filter, setFilter] = useState('all');
  const [notes, setNotes] = useState('');
  const [notesEditing, setNotesEditing] = useState(true);
  const [showNotesSavedModal, setShowNotesSavedModal] = useState(false);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [videoMode, setVideoMode] = useState('clip');
  const [videoTime, setVideoTime] = useState(0);
  const [focusMode, setFocusMode] = useState(false);
  const [storyboard, setStoryboard] = useState([]);
  const [loadingStoryboard, setLoadingStoryboard] = useState(false);
  const videoRef = useRef(null);
  const timelineRef = useRef(null);
  const timelineTimeRef = useRef(null);
  const lastPositionSave = useRef(0);
  const pendingStatePatch = useRef({});
  const stateSaveTimer = useRef(null);
  const seekTelemetryTimer = useRef(null);
  const downloadProgressUnsubscribe = useRef(null);

  const refreshSessions = useCallback(async () => {
    try {
      const response = await fetch(`${API}/sessions`);
      if (!response.ok) throw new Error();
      setSessions(await response.json());
    } catch {
      setSessions([]);
    }
  }, []);

  const refreshClips = useCallback(async (id = sessionId) => {
    const response = await fetch(`${API}/sessions/${id}/clips`);
    if (!response.ok) throw new Error('Não foi possível carregar os clipes da análise.');
    return response.json();
  }, [sessionId]);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);
  useEffect(() => {
    if (typeof window.desktop?.onDownloadProgress !== 'function') {
      setError('A integração com o aplicativo não carregou. Feche e abra o aplicativo novamente; se persistir, consulte os logs do Electron.');
      return undefined;
    }
    downloadProgressUnsubscribe.current = window.desktop.onDownloadProgress((progress) => {
      setCompilationProgress((current) => current?.downloadId === progress.downloadId
        ? { ...current, receivedBytes: progress.receivedBytes, totalBytes: progress.totalBytes, downloadState: progress.state }
        : current);
    });
    return () => downloadProgressUnsubscribe.current?.();
  }, []);
  useEffect(() => {
    if (!sessionId) return undefined;
    logTelemetry('review_started');
    return () => logTelemetry('review_finished');
  }, [sessionId]);

  const startAnalysis = async () => {
    if (!videoPath) return;
    setAnalysisErrorDialog(null);
    setError(''); setNotice(''); setClips([]); setCurrentClip(null); setNotes(''); setNotesEditing(true); setSelected(new Set());
    setStatus({ status: 'starting', stage: 'Iniciando análise', progress: 0 });
    try {
      const response = await fetch(`${API}/sessions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoPath, mode }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível iniciar a análise.');
      setSessionId(payload.session_id);
    } catch (caught) {
      setStatus(null);
      const detail = caught.message || 'Não foi possível conectar à API local. Feche e inicie novamente o aplicativo.';
      setError(detail);
      setAnalysisErrorDialog({ title: 'Não foi possível iniciar a análise', message: 'Confira o vídeo selecionado e a conexão com o serviço local antes de tentar novamente.', detail });
    }
  };

  const cancelAnalysis = async () => {
    if (!sessionId || !['running', 'starting'].includes(status?.status)) return;
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/cancel`, { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível cancelar a análise.');
      setStatus((current) => ({ ...current, status: payload.status, stage: 'Cancelando análise' }));
      setNotice('Cancelamento solicitado. Os arquivos parciais serão descartados quando o processamento parar.');
    } catch (caught) {
      setError(caught.message || 'Não foi possível cancelar a análise.');
    }
  };

  const openSession = async (id) => {
    setAnalysisErrorDialog(null);
    setError(''); setNotice(''); setSessionId(''); setStatus(null); setClips([]); setCurrentClip(null);
    try {
      const response = await fetch(`${API}/sessions/${id}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível abrir esta análise.');
      setStatus(payload);
      setFilter(payload.review_filter || 'all');
      setAutoAdvance(Boolean(payload.auto_advance));
      setVideoMode(payload.video_mode || 'clip');
      setVideoTime(Number(payload.current_time || 0));
      setSessionId(id);
    } catch (caught) {
      setError(caught.message || 'Não foi possível abrir esta análise.');
      refreshSessions();
    }
  };

  useEffect(() => {
    if (!sessionId) return undefined;
    let active = true;
    let timer;
    const poll = async () => {
      try {
        const response = await fetch(`${API}/sessions/${sessionId}`);
        const next = await response.json();
        if (!response.ok) throw new Error(next.detail || 'A conexão com a análise foi perdida.');
        if (!active) return;
        setStatus(next);
        if (next.status === 'completed') {
          const items = await refreshClips(sessionId);
          if (!active) return;
          setClips(items);
          const selectedClip = items.find((item) => item.segment_id === next.current_segment_id) || items[0] || null;
          setCurrentClip(selectedClip);
          setNotes(selectedClip?.notes || '');
          setNotesEditing(!selectedClip?.notes);
          return;
        }
        if (next.status === 'failed') {
          const detail = next.error || 'A análise não foi concluída.';
          setError(detail);
          setAnalysisErrorDialog({ title: 'Não foi possível concluir a análise', message: 'Confira se o arquivo de vídeo é válido e tente iniciar uma nova análise.', detail });
          return;
        }
        if (next.status === 'cancelled') {
          setNotice('Análise cancelada. Os arquivos parciais foram descartados; as sessões concluídas continuam no histórico.');
          await refreshSessions();
          return;
        }
        if (next.status === 'interrupted') return;
        timer = setTimeout(poll, 700);
      } catch (caught) {
        if (!active) return;
        setError(caught.message || 'A conexão com a análise foi perdida.');
        timer = setTimeout(poll, 1500);
      }
    };
    poll();
    return () => { active = false; clearTimeout(timer); };
  }, [sessionId, refreshClips]);

  const logTelemetry = useCallback((event, candidateId = currentClip?.candidate_id, value) => {
    if (!sessionId) return;
    fetch(`${API}/sessions/${sessionId}/telemetry`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, candidate_id: candidateId, value }), keepalive: event === 'review_finished',
    }).catch(() => {});
  }, [sessionId, currentClip?.candidate_id]);



  const saveSessionState = useCallback((patch, immediate = false) => {
    if (!sessionId) return;
    pendingStatePatch.current = { ...pendingStatePatch.current, ...patch };
    if (stateSaveTimer.current) clearTimeout(stateSaveTimer.current);
    const flush = async () => {
      const body = pendingStatePatch.current;
      pendingStatePatch.current = {};
      stateSaveTimer.current = null;
      if (!Object.keys(body).length) return;
      try {
        await fetch(`${API}/sessions/${sessionId}/state`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
      } catch { /* Uma próxima alteração persistirá o estado novamente. */ }
    };
    stateSaveTimer.current = setTimeout(flush, immediate ? 0 : 600);
  }, [sessionId]);

  const selectClip = (clip) => {
    const seekTime = videoMode === 'source' ? Number(clip?.timestamp_seconds || clip?.start_seconds || 0) : 0;
    setCurrentClip(clip); setNotes(clip?.notes || ''); setNotesEditing(!clip?.notes); setVideoTime(seekTime);
    setStoryboard([]);
    logTelemetry('candidate_opened', clip.candidate_id);
    saveSessionState({ current_segment_id: clip.segment_id, current_time: seekTime });
    if (videoMode === 'source' && videoRef.current) videoRef.current.currentTime = seekTime;
    if (timelineRef.current && videoMode === 'source') timelineRef.current.value = String(seekTime);
    if (timelineTimeRef.current && videoMode === 'source') timelineTimeRef.current.textContent = formatDuration(seekTime);
  };
  const currentIndex = clips.findIndex((clip) => clip.segment_id === currentClip?.segment_id);
  const navigateClip = useCallback((delta) => {
    const index = clips.findIndex((clip) => clip.segment_id === currentClip?.segment_id);
    const next = clips[index + delta];
    if (next) selectClip(next);
  }, [clips, currentClip, saveSessionState]);

  const updateReview = async (patch, optimistic = {}) => {
    if (!currentClip) return false;
    const segmentId = currentClip.segment_id;
    setSavingClips((items) => new Set(items).add(segmentId));
    setClips((items) => items.map((item) => item.segment_id === segmentId ? { ...item, ...optimistic } : item));
    setCurrentClip((item) => item?.segment_id === segmentId ? { ...item, ...optimistic } : item);
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/clips/${segmentId}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível salvar a revisão.');
      if (patch.tag) logTelemetry('decision_saved', currentClip.candidate_id, patch.tag);
      if (patch.tag) setStatus((current) => current ? { ...current, can_undo: payload.can_undo } : current);
      if (patch.review_status === 'deferred') logTelemetry('review_deferred', currentClip.candidate_id);
      if (patch.review_status === 'deferred') setNotice('Candidato marcado para revisar depois. Ele foi movido para o filtro “Revisar depois”.');
      if (patch.review_status === 'pending') setNotice('Candidato voltou para a lista de pendentes.');
      setSavingClips((items) => { const next = new Set(items); next.delete(segmentId); return next; });
      await refreshSessions();
      return true;
    } catch (caught) {
      setSavingClips((items) => { const next = new Set(items); next.delete(segmentId); return next; });
      setError(caught.message || 'Não foi possível salvar a revisão.');
      const items = await refreshClips(sessionId); setClips(items);
      setCurrentClip(items.find((item) => item.segment_id === segmentId) || null);
      return false;
    }
  };

  const markClip = async (tag) => {
    const saved = await updateReview({ tag }, { human_label: tag, annotation_status: 'completed' });
    if (saved && autoAdvance) navigateClip(1);
  };
  const undoLastDecision = async () => {
    if (!sessionId || !status?.can_undo || undoingDecision) return;
    setUndoingDecision(true);
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/undo`, { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível desfazer a última decisão.');
      const items = await refreshClips(sessionId);
      setClips(items);
      setStatus((current) => current ? { ...current, can_undo: payload.can_undo } : current);
      const restoredClip = items.find((item) => item.segment_id === payload.segment_id);
      if (restoredClip) {
        setCurrentClip(restoredClip);
        setNotes(restoredClip.notes || '');
        setNotesEditing(!restoredClip.notes);
        const seekTime = videoMode === 'source' ? Number(restoredClip.timestamp_seconds || restoredClip.start_seconds || 0) : 0;
        setVideoTime(seekTime);
        saveSessionState({ current_segment_id: restoredClip.segment_id, current_time: seekTime }, true);
      }
      setNotice('Última decisão desfeita. O candidato voltou ao estado anterior.');
      await refreshSessions();
    } catch (caught) {
      setError(caught.message || 'Não foi possível desfazer a última decisão.');
    } finally {
      setUndoingDecision(false);
    }
  };
  const deferClip = async () => {
    const saved = await updateReview({ review_status: 'deferred' }, { annotation_status: 'deferred' });
    if (saved && autoAdvance) navigateClip(1);
  };
  const returnToPending = () => updateReview({ review_status: 'pending' }, { annotation_status: 'pending' });
  const saveNotes = async () => {
    const saved = await updateReview({ notes }, { notes });
    if (saved) {
      setNotesEditing(false);
      setShowNotesSavedModal(true);
    }
  };

  const downloadUrl = (clip) => `${API}/sessions/${sessionId}/clips/${clip.segment_id}/download`;
  const downloadClip = async (clip) => {
    setPreparingDownloads((items) => new Set(items).add(clip.segment_id));
    try { await window.desktop.download(downloadUrl(clip)); }
    catch { setError('Não foi possível preparar o download do clipe.'); }
    finally { setPreparingDownloads((items) => { const next = new Set(items); next.delete(clip.segment_id); return next; }); }
  };
  const downloadSelected = async () => {
    if (!selected.size) return;
    setPreparingCompilation(true);
    setError('');
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/compilations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ segment_ids: [...selected] }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível iniciar a compilação.');
      const jobId = payload.job_id;
      setCompilationProgress({ jobId, stage: 'Preparando clipes', progress: 0, phase: 'compile' });
      let result;
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        const statusResponse = await fetch(`${API}/sessions/${sessionId}/compilations/${jobId}`);
        result = await statusResponse.json();
        if (!statusResponse.ok) throw new Error(result.detail || 'Não foi possível acompanhar a compilação.');
        setCompilationProgress((current) => current?.jobId === jobId ? { ...current, ...result, phase: 'compile' } : current);
        if (result.status === 'ready') break;
        if (result.status === 'failed') throw new Error(result.error || 'A compilação falhou.');
        if (result.status === 'cancelled') return;
      }
      const downloadId = `compilation-${Date.now()}`;
      setCompilationProgress((current) => ({ ...current, stage: 'Baixando o vídeo compilado', phase: 'download', progress: 0, downloadId }));
      await window.desktop.download(`${API}/sessions/${sessionId}/compilations/${jobId}/download`, downloadId);
      setNotice('Compilação baixada.');
    } catch (caught) {
      if (caught.message !== 'Download cancelado.') setError(caught.message || 'Não foi possível preparar a compilação.');
    } finally {
      setPreparingCompilation(false);
      setCompilationProgress(null);
    }
  };
  const cancelCompilation = async () => {
    const active = compilationProgress;
    if (!active) return;
    if (active.phase === 'download' && active.downloadId) await window.desktop.cancelDownload(active.downloadId);
    if (active.jobId) await fetch(`${API}/sessions/${sessionId}/compilations/${active.jobId}`, { method: 'DELETE' }).catch(() => {});
    setCompilationProgress(null);
    setPreparingCompilation(false);
    setNotice('Compilação cancelada.');
  };

  const removeSession = async (event, item) => {
    event.stopPropagation();
    if (!window.confirm(`Remover a análise de “${item.video_name || 'Vídeo'}” e os arquivos gerados por ela? O vídeo original não será apagado.`)) return;
    try {
      const response = await fetch(`${API}/sessions/${item.session_id}`, { method: 'DELETE' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível remover esta análise.');
      if (sessionId === item.session_id) { setSessionId(''); setStatus(null); setClips([]); setCurrentClip(null); }
      await refreshSessions();
      setNotice('Análise removida do histórico.');
    } catch (caught) { setError(caught.message || 'Não foi possível remover esta análise.'); }
  };
  const toggleSelection = (id) => setSelected((previous) => { const next = new Set(previous); next.has(id) ? next.delete(id) : next.add(id); return next; });
  const exportReview = async (format) => {
    setPreparingExport(format);
    try {
      const url = format === 'html' ? `${API}/sessions/${sessionId}/report.html` : `${API}/sessions/${sessionId}/export?format=${format}`;
      await window.desktop.download(url); logTelemetry('export', currentClip?.candidate_id, format);
    }
    catch { setError(`Não foi possível exportar a revisão em ${format.toUpperCase()}.`); }
    finally { setPreparingExport(''); }
  };

  const setRate = useCallback((rate) => {
    const normalized = Math.max(0.25, Math.min(3, Number(rate.toFixed(2))));
    setPlaybackRate(normalized);
    if (videoRef.current) videoRef.current.playbackRate = normalized;
    logTelemetry('rate_change', currentClip?.candidate_id, normalized);
  }, [logTelemetry, currentClip?.candidate_id]);
  const stepRate = useCallback((direction) => {
    const rates = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3];
    const index = rates.reduce((best, rate, candidate) => Math.abs(rate - playbackRate) < Math.abs(rates[best] - playbackRate) ? candidate : best, 0);
    setRate(rates[Math.max(0, Math.min(rates.length - 1, index + direction))]);
  }, [playbackRate, setRate]);

  const switchVideoMode = () => {
    if (!currentClip) return;
    setError('');
    const nextMode = videoMode === 'clip' ? 'source' : 'clip';
    const seekTime = nextMode === 'source' ? Number(currentClip.timestamp_seconds || currentClip.start_seconds || 0) : 0;
    setVideoMode(nextMode); setVideoTime(seekTime);
    saveSessionState({ video_mode: nextMode, current_segment_id: currentClip.segment_id, current_time: seekTime });
  };

  const loadStoryboard = async () => {
    if (!currentClip) return;
    setLoadingStoryboard(true); setError('');
    try {
      const response = await fetch(`${API}/sessions/${sessionId}/clips/${currentClip.segment_id}/storyboard`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Não foi possível gerar o storyboard.');
      setStoryboard(payload.map((frame) => ({ ...frame, image_url: `${API}${frame.image_url}` })));
      logTelemetry('storyboard_used', currentClip.candidate_id);
    } catch (caught) { setError(caught.message || 'Não foi possível gerar o storyboard.'); }
    finally { setLoadingStoryboard(false); }
  };

  useEffect(() => {
    if (!currentClip || !videoRef.current || !videoTime) return;
    const player = videoRef.current;
    const restore = () => { player.currentTime = Math.min(videoTime, player.duration || videoTime); player.playbackRate = playbackRate; };
    if (player.readyState >= 1) restore();
    else player.addEventListener('loadedmetadata', restore, { once: true });
    return () => player.removeEventListener('loadedmetadata', restore);
  }, [currentClip?.segment_id, videoMode]);

  useEffect(() => {
    saveSessionState({ review_filter: filter, auto_advance: autoAdvance });
  }, [filter, autoAdvance, saveSessionState]);

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (!sessionId || !currentClip) return;
      if (event.repeat) return;
      const player = videoRef.current;
      const textEntryFocused = target instanceof HTMLElement && (['INPUT', 'TEXTAREA'].includes(target.tagName) || target.isContentEditable);
      if (!textEntryFocused && (event.key === '[' || event.key === ']')) {
        event.preventDefault(); event.stopPropagation(); stepRate(event.key === '[' ? -1 : 1); return;
      }
      if (target instanceof HTMLElement && (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'SUMMARY'].includes(target.tagName) || target.isContentEditable)) return;
      if (event.code === 'Space') { event.preventDefault(); event.stopPropagation(); if (player?.paused) player.play(); else player?.pause(); }
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault(); event.stopPropagation(); if (player) player.currentTime = Math.max(0, player.currentTime + (event.key === 'ArrowLeft' ? -1 : 1) * (event.ctrlKey ? 10 : event.shiftKey ? 1 : 5));
      } else if (event.key.toLowerCase() === 'a') navigateClip(-1);
      else if (event.key.toLowerCase() === 'd') navigateClip(1);
      else if (event.key === ',') { event.preventDefault(); const step = 1 / Number(status?.video_fps || 30); if (player) player.currentTime = Math.max(0, player.currentTime - step); setNotice(`Avanço aproximado de um frame (${step.toFixed(4)} s).`); }
      else if (event.key === '.') { event.preventDefault(); const step = 1 / Number(status?.video_fps || 30); if (player) player.currentTime += step; setNotice(`Avanço aproximado de um frame (${step.toFixed(4)} s).`); }
      else if (event.key === '1') markClip('GOAL');
      else if (event.key === '2') markClip('NO_GOAL');
      else if (event.key === '3') markClip('NOT_EVALUABLE');
      else if (event.key.toLowerCase() === 'r') deferClip();
      else if (event.key.toLowerCase() === 'f') setFocusMode((value) => { logTelemetry('focus_mode', currentClip.candidate_id, !value); return !value; });
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [sessionId, status?.video_fps, currentClip, navigateClip, markClip, deferClip, stepRate, logTelemetry]);

  const onTimeUpdate = () => {
    const value = videoRef.current?.currentTime || 0;
    if (timelineRef.current) timelineRef.current.value = String(value);
    if (timelineTimeRef.current) timelineTimeRef.current.textContent = formatDuration(value);
    if (Date.now() - lastPositionSave.current > 5000) {
      lastPositionSave.current = Date.now();
      saveSessionState({ current_segment_id: currentClip?.segment_id, current_time: value });
    }
  };
  const toggleFocus = () => {
    setFocusMode((value) => !value);
    logTelemetry('focus_mode', currentClip?.candidate_id, !focusMode);
  };

  const visibleClips = clips.filter((clip) => filter === 'all' || (clip.annotation_status || 'pending') === filter);
  const currentCandidateNumber = candidateNumber(Math.max(0, currentIndex));
  const metrics = status?.metrics;
  const compiledDownloadPercent = compilationProgress?.phase === 'download' && compilationProgress.totalBytes > 0
    ? Math.min(100, compilationProgress.receivedBytes / compilationProgress.totalBytes * 100)
    : Number(compilationProgress?.progress || 0);
  const showDashboard = !sessionId;

  return <main className={`${dark ? 'dark' : ''} ${focusMode ? 'focus-mode' : ''}`}>
    <header><div><p>Revisão assistida de candidatos</p><h1>Goal Event Review</h1></div><div className="header-actions">{sessionId && <button onClick={() => { setSessionId(''); setStatus(null); setClips([]); setCurrentClip(null); setError(''); refreshSessions(); }}><ArrowLeft />Histórico</button>}<button className="icon" onClick={() => setDark(!dark)} title="Alternar tema">{dark ? <Sun /> : <Moon />}</button></div></header>
    {error && !analysisErrorDialog && <p className="notice error" role="alert">{error}</p>}
    {notice && <p className="notice" role="status">{notice}</p>}

    {showDashboard && <>
      <section className="setup"><input readOnly value={videoPath} placeholder="Selecione um vídeo" aria-label="Vídeo selecionado" /><button onClick={async () => setVideoPath(await window.desktop.selectVideo())}><FolderOpen />Selecionar vídeo</button><select value={mode} onChange={(event) => setMode(event.target.value)} aria-label="Modo operacional"><option value="alta_cobertura">Alta cobertura</option><option value="cobertura_reduzida">Cobertura reduzida</option><option value="triagem_rapida">Triagem rápida</option></select><button onClick={startAnalysis} disabled={!videoPath}><Play />Nova análise</button></section>
      <section><div className="listhead"><div><h2>Análises recentes</h2><small>{sessions.length} análise(s) local(is)</small></div></div>
        {sessions.length === 0 ? <p className="muted">As análises que você iniciar ficarão disponíveis aqui para retomada.</p> : <div className="session-list">{sessions.map((item) => <div className="session-card" key={item.session_id} role="button" tabIndex={0} onClick={() => openSession(item.session_id)} onKeyDown={(event) => { if (event.key === 'Enter' && event.target === event.currentTarget) openSession(item.session_id); }}><span><b>{item.video_name || 'Vídeo'}</b><small>{modeText(item.mode)} · {item.created_at ? new Date(item.created_at).toLocaleString() : 'Data indisponível'}</small><small>Duração da análise: {item.elapsed ? formatDuration(item.elapsed) : '—'} · Equipamento: {item.metrics?.device_name || item.metrics?.cpu_name || '—'}</small></span><span className={`session-status ${item.status}`}>{item.status === 'completed' ? 'Concluída' : item.status === 'running' ? 'Processando' : item.status === 'cancelling' ? 'Cancelando' : item.status === 'cancelled' ? 'Cancelada' : item.status === 'interrupted' ? 'Interrompida' : item.status === 'failed' ? 'Falhou' : item.status}<small>{item.reviewed_count || 0} decisões</small></span><button className="icon small remove-session" title="Remover análise do histórico" aria-label={`Remover análise ${item.video_name || ''}`} disabled={['running', 'cancelling'].includes(item.status)} onClick={(event) => removeSession(event, item)}><Trash2 /></button></div>)}</div>}
      </section>
    </>}

    {sessionId && <>
      <section className="session-heading"><div><p>Análise local</p><h2>{status?.video_name || 'Vídeo'}</h2><small>Modo: {modeText(status?.mode)} · {status?.created_at ? new Date(status.created_at).toLocaleString() : ''}</small><details className="technical-details"><summary>Detalhes do processamento</summary><div><small>Tempo total: {formatDuration(status?.elapsed)}<br />Equipamento: {metrics?.device_name || metrics?.cpu_name || 'Não informado'} ({metrics?.processor || '—'})<br />CPU usada pelo processo: {metrics?.process_cpu_percent ?? '—'}% · RAM do processo: {metrics?.process_ram_mb ?? '—'} MB ({metrics?.process_ram_percent ?? '—'}%)<br />Uso de GPU: {metrics?.gpu_utilization_percent != null ? `${metrics.gpu_utilization_percent}%` : 'Indisponível neste ambiente'}<br />Memória da GPU: {metrics?.gpu_memory_used_mb != null ? `${metrics.gpu_memory_used_mb} / ${metrics.gpu_memory_total_mb} MB (${metrics.gpu_memory_percent}%)` : 'Não disponível'}<br />Aplicativo {status?.app_version} · Pipeline {status?.pipeline_version}<br /><span>As impressões SHA-256 ficam nos arquivos de auditoria para identificar os modelos e a configuração usados.</span></small></div></details></div><div className="export-actions"><span className="export-help">Relatório: resumo legível · CSV/JSON: dados detalhados</span><button disabled={preparingExport === 'html'} onClick={() => exportReview('html')}><Download />{preparingExport === 'html' ? 'Preparando relatório…' : 'Relatório legível'}</button><button disabled={preparingExport === 'csv'} onClick={() => exportReview('csv')}><Download />{preparingExport === 'csv' ? 'Preparando CSV…' : 'Dados em CSV'}</button><button disabled={preparingExport === 'json'} onClick={() => exportReview('json')}><Download />{preparingExport === 'json' ? 'Preparando JSON…' : 'Dados em JSON'}</button></div></section>
      {status && ['running', 'starting', 'cancelling'].includes(status.status) && <section className="progress"><div className="progress-top"><div><b>{status.stage}</b><small>{status.status === 'cancelling' ? 'Interrompendo o processamento e limpando os arquivos parciais' : progressText(status)}</small></div><strong>{Number(status.progress || 0).toFixed(2).replace('.', ',')}%</strong></div><div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(Number(status.progress || 0), 100))}%` }} /></div><button className="cancel-analysis-button" disabled={status.status === 'cancelling'} onClick={cancelAnalysis}>{status.status === 'cancelling' ? 'Cancelando…' : 'Cancelar análise'}</button></section>}
      {metrics && <section className="metrics"><div><span><b>Processador</b><small>{metrics.processor === 'GPU' ? metrics.device_name : metrics.cpu_name} · {metrics.processor}</small></span></div><div><span><b>CPU do processo</b><small>{metrics.process_cpu_percent}%</small></span></div><div><span><b>Memória RAM</b><small>{metrics.process_ram_mb} MB · {metrics.process_ram_percent}%</small></span></div><div><span><b>Uso de GPU</b><small>{metrics.gpu_utilization_percent != null ? `${metrics.gpu_utilization_percent}%` : 'Indisponível neste ambiente'}</small></span></div><div><span><b>Memória da GPU</b><small>{metrics.gpu_memory_used_mb != null ? `${metrics.gpu_memory_used_mb} / ${metrics.gpu_memory_total_mb} MB · ${metrics.gpu_memory_percent}%` : 'Indisponível'}</small></span></div><div><span><b>Tempo de processamento</b><small>{formatDuration(status?.elapsed)}</small></span></div></section>}
      {status?.status === 'interrupted' && <section className="empty compact"><Video /><h2>Análise interrompida</h2><p>As decisões já persistidas continuam no histórico. Para processar o vídeo novamente, inicie uma nova análise.</p></section>}
      {status?.status === 'cancelled' && <section className="empty compact"><Video /><h2>Análise cancelada</h2><p>A saída parcial foi removida. Inicie uma nova análise se quiser processar este vídeo novamente.</p></section>}
      {status?.status === 'failed' && <section className="empty compact"><Video /><h2>Não foi possível concluir a análise</h2><p>Confira o aviso de erro e corrija o arquivo antes de iniciar outra análise.</p></section>}
      {status?.status === 'completed' && clips.length === 0 && <section className="empty compact"><Video /><h2>Nenhum clipe disponível</h2><p>O resultado desta análise não contém clipes para revisar.</p></section>}
      {clips.length > 0 && <section className="workspace">
        <aside><div className="listhead"><div><h2>Candidatos</h2><small>{selected.size ? `${selected.size} selecionado(s)` : `${visibleClips.length} de ${clips.length}`}</small><small className="list-order-help">Ordenados pelo ranking do algoritmo.</small></div><button className="download-selected" disabled={preparingCompilation || !selected.size} onClick={downloadSelected}>{preparingCompilation ? 'Preparando…' : 'Compilar selecionados'}</button></div>
          <div className="filter-row"><label htmlFor="review-filter">Estado</label><select id="review-filter" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">Todos</option><option value="pending">Pendentes</option><option value="deferred">Revisar depois</option><option value="completed">Concluídos</option></select><label className="auto-advance"><input type="checkbox" checked={autoAdvance} onChange={(event) => setAutoAdvance(event.target.checked)} />Avançar após decisão</label></div>
          <div className="clip-list">{visibleClips.map((clip) => <div className={`clip ${currentClip?.segment_id === clip.segment_id ? 'active' : ''}`} onClick={() => selectClip(clip)} key={clip.segment_id} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter') selectClip(clip); }}>
            <input type="checkbox" aria-label={`Selecionar candidato ${candidateNumber(clips.indexOf(clip))} para compilação`} checked={selected.has(clip.segment_id)} onClick={(event) => event.stopPropagation()} onChange={() => toggleSelection(clip.segment_id)} />
            <span className={`tag ${clip.human_label || 'UNREVIEWED'}`}>{clip.human_label || 'Sem decisão'}</span><b>Candidato {candidateNumber(clips.indexOf(clip))}</b>
            <small>{clip.timestamp || formatDuration(clip.timestamp_seconds)} &middot; {reviewText(clip.annotation_status)}</small>
          </div>)}</div>
        </aside>
        <article>{currentClip && <>
          <div className="player"><video ref={videoRef} key={videoMode === 'source' ? 'full-source' : `${currentClip.segment_id}-${currentClip.clip_filename}`} controls src={videoMode === 'source' ? `${API}/sessions/${sessionId}/source` : `${API}/sessions/${sessionId}/clips/${currentClip.segment_id}/media`} onTimeUpdate={onTimeUpdate} onPlay={() => logTelemetry('play')} onPause={() => { logTelemetry('pause'); saveSessionState({ current_segment_id: currentClip.segment_id, current_time: videoRef.current?.currentTime || 0 }, true); }} onSeeked={() => { const value = videoRef.current?.currentTime || 0; saveSessionState({ current_segment_id: currentClip.segment_id, current_time: value }, true); if (seekTelemetryTimer.current) clearTimeout(seekTelemetryTimer.current); seekTelemetryTimer.current = setTimeout(() => logTelemetry('seek', currentClip.candidate_id, value), 600); }} onLoadedMetadata={() => { if (videoRef.current) { videoRef.current.playbackRate = playbackRate; videoRef.current.currentTime = Math.min(videoTime, videoRef.current.duration || videoTime); if (videoMode === 'source' && Number.isFinite(videoRef.current.duration)) setStatus((value) => ({ ...value, video_duration_seconds: videoRef.current.duration })); } if (timelineRef.current) timelineRef.current.value = String(videoTime); }} onError={() => setError(videoMode === 'source' ? 'Este navegador não conseguiu reproduzir o vídeo original. Volte ao modo clipe para continuar a revisão.' : 'Não foi possível reproduzir este clipe. Verifique se o arquivo está disponível e em formato compatível.')} /></div>
          <div className="clip-details"><div><p>Detalhes do candidato</p><h2>Candidato {currentCandidateNumber}</h2><small>ID: {currentClip.candidate_id || currentClip.segment_id}</small></div><button className="icon small" title="Baixar clipe" disabled={preparingDownloads.has(currentClip.segment_id) || savingClips.has(currentClip.segment_id)} onClick={() => downloadClip(currentClip)}><Download /></button></div>
          <div className="candidate-meta"><div><small>Timestamp principal</small><b>{currentClip.timestamp || formatDuration(currentClip.timestamp_seconds)}</b></div><div><small>Posição no ranking</small><b>{currentClip.rank || currentIndex + 1}</b></div><div><small>Rank score</small><b>{currentClip.rank_score || '—'}</b></div><div><small>Score</small><b>{currentClip.score || '—'}</b></div><div><small>Método</small><b>{currentClip.method || 'Não informado'}</b></div><div><small>Trecho incluído</small><b>{formatDuration(currentClip.start_seconds)}–{formatDuration(currentClip.end_seconds)}</b></div></div>
          <div className="player-actions"><button disabled={currentIndex <= 0} onClick={() => navigateClip(-1)}><ArrowLeft />Anterior <kbd>A</kbd></button><span>{currentIndex + 1} / {clips.length}</span><button disabled={currentIndex >= clips.length - 1} onClick={() => navigateClip(1)}>Próximo <kbd>D</kbd><ArrowRight /></button><label>Velocidade<select value={playbackRate} onChange={(event) => setRate(Number(event.target.value))}>{[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3].map((rate) => <option key={rate} value={rate}>{rate}×</option>)}</select></label><details className="shortcut-help"><summary title="Atalhos do teclado" aria-label="Mostrar atalhos do teclado"><CircleHelp /></summary><div className="shortcut-popover"><b>Atalhos do teclado</b><p><kbd>Espaço</kbd> reproduzir/pausar &middot; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> 5 s &middot; <kbd>Shift</kbd> + setas 1 s &middot; <kbd>Ctrl</kbd> + setas 10 s &middot; <kbd>,</kbd>/<kbd>.</kbd> um frame &middot; <kbd>[</kbd>/<kbd>]</kbd> velocidade &middot; <kbd>A</kbd>/<kbd>D</kbd> candidato anterior/próximo &middot; <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> decisão &middot; <kbd>R</kbd> revisar depois &middot; <kbd>F</kbd> modo de foco.</p></div></details><button onClick={switchVideoMode}>{videoMode === 'source' ? 'Reproduzir clipe' : 'Partida completa'}</button><button onClick={loadStoryboard} disabled={loadingStoryboard}>{loadingStoryboard ? 'Gerando…' : 'Storyboard'}</button><button onClick={toggleFocus}>{focusMode ? 'Sair do foco' : 'Modo de foco'} <kbd>F</kbd></button></div>
          {videoMode === 'source' && <div className="timeline"><p className="timeline-help">Partida completa: as faixas verdes mostram onde estão os clipes. Clique numa faixa para abrir esse candidato; arraste o controle para navegar pelo vídeo.</p><div className="timeline-input"><input ref={timelineRef} type="range" min="0" max={status?.video_duration_seconds || 1} step="0.1" defaultValue={videoTime} aria-label="Posição na partida completa" onPointerUp={() => saveSessionState({ current_segment_id: currentClip.segment_id, current_time: videoRef.current?.currentTime || videoTime }, true)} onChange={(event) => { const value = Number(event.target.value); if (videoRef.current) videoRef.current.currentTime = value; setVideoTime(value); }} /><div className="timeline-markers">{clips.map((clip, clipIndex) => { const total = Number(status?.video_duration_seconds || 1); const start = Number(clip.start_seconds || 0); const end = Number(clip.end_seconds || start); return <button key={clip.segment_id} className={clip.segment_id === currentClip.segment_id ? 'active' : ''} style={{ left: `${Math.max(0, Math.min(100, start / total * 100))}%`, width: `${Math.max(0.25, Math.min(100, (end - start) / total * 100))}%` }} title={`Candidato ${candidateNumber(clipIndex)} · ${clip.timestamp || formatDuration(clip.timestamp_seconds)}`} aria-label={`Ir para candidato ${candidateNumber(clipIndex)}`} onClick={() => selectClip(clip)} />; })}</div></div><div className="timeline-legend"><span><i className="legend-clip" />Intervalo de clipe candidato</span><span><i className="legend-current" />Candidato selecionado</span></div><div className="timeline-labels"><span ref={timelineTimeRef}>{formatDuration(videoTime)}</span><span>{formatDuration(status?.video_duration_seconds)}</span></div></div>}
          {storyboard.length > 0 && <><p className="storyboard-help">Storyboard: cinco quadros próximos ao momento do candidato para dar contexto visual. Clique em um quadro para ir até aquele instante.</p><div className="storyboard" aria-label="Storyboard do candidato">{storyboard.map((frame) => <button key={frame.offset_seconds} title={`Ir para ${formatDuration(frame.timestamp_seconds)} no vídeo`} onClick={() => { const target = videoMode === 'source' ? frame.timestamp_seconds : frame.clip_time_seconds; if (videoRef.current) videoRef.current.currentTime = target; setVideoTime(target); }}><img src={frame.image_url} alt={`Quadro em T${frame.offset_seconds >= 0 ? '+' : ''}${frame.offset_seconds} segundos`} /><small>T{frame.offset_seconds >= 0 ? '+' : ''}{frame.offset_seconds}s</small></button>)}</div></>}
          <div className="tags"><div><b>Decisão humana</b><small>O score é uma ordenação operacional, não uma probabilidade de gol.</small></div><div className="tag-buttons">{TAGS.map((tag) => <button disabled={savingClips.has(currentClip.segment_id)} className={`${tag} ${currentClip.human_label === tag ? 'selected-tag' : ''}`} onClick={() => markClip(tag)} key={tag}>{tag}</button>)}<button className="defer-button" disabled={savingClips.has(currentClip.segment_id) || Boolean(currentClip.human_label) || currentClip.annotation_status === 'deferred'} onClick={deferClip}>{currentClip.annotation_status === 'deferred' ? 'Adiado ✓' : 'Revisar depois'}</button>{currentClip.annotation_status === 'deferred' && <button onClick={returnToPending} disabled={savingClips.has(currentClip.segment_id)}>Voltar a pendente</button>}{status?.can_undo && <button className="undo-decision-button" onClick={undoLastDecision} disabled={undoingDecision || savingClips.size > 0}>{undoingDecision ? 'Desfazendo…' : 'Desfazer última decisão'}</button>}</div></div>
          <div className="notes"><label htmlFor="clip-notes">Notas da revisão</label><textarea id="clip-notes" rows="3" value={notes} readOnly={!notesEditing} onChange={(event) => setNotes(event.target.value)} placeholder="Observações humanas opcionais" /><button onClick={() => { if (notesEditing) saveNotes(); else setNotesEditing(true); }} disabled={savingClips.has(currentClip.segment_id)}>{notesEditing ? 'Salvar nota' : 'Editar nota'}</button><small>Estado atual: {reviewText(currentClip.annotation_status)}</small></div>
        </>}</article>
      </section>}
    </>}
    {showNotesSavedModal && <div className="modal-backdrop note-modal-backdrop" role="presentation"><section className="progress-modal note-saved-modal" role="dialog" aria-modal="true" aria-labelledby="note-saved-title" aria-describedby="note-saved-description" onKeyDown={(event) => { if (event.key === 'Escape') setShowNotesSavedModal(false); }}><CheckCircle2 className="note-saved-icon" aria-hidden="true" /><h2 id="note-saved-title">Nota salva</h2><p id="note-saved-description">A nota foi salva para este candidato.</p><button autoFocus onClick={() => setShowNotesSavedModal(false)}>Entendi</button></section></div>}
    {analysisErrorDialog && <div className="modal-backdrop note-modal-backdrop" role="presentation"><section className="progress-modal note-saved-modal analysis-error-modal" role="dialog" aria-modal="true" aria-labelledby="analysis-error-title" aria-describedby="analysis-error-description" onKeyDown={(event) => { if (event.key === 'Escape') setAnalysisErrorDialog(null); }}><CircleAlert className="analysis-error-icon" aria-hidden="true" /><h2 id="analysis-error-title">{analysisErrorDialog.title}</h2><p id="analysis-error-description">{analysisErrorDialog.message}</p>{analysisErrorDialog.detail && <details className="analysis-error-details"><summary>Detalhes técnicos</summary><pre>{analysisErrorDialog.detail}</pre></details>}<button autoFocus onClick={() => setAnalysisErrorDialog(null)}>Entendi</button></section></div>}
    {compilationProgress && <div className="modal-backdrop" role="presentation"><section className="progress-modal" role="dialog" aria-modal="true" aria-labelledby="compilation-title"><h2 id="compilation-title">Compilando e baixando clipes</h2><p>{compilationProgress.phase === 'download' ? compilationProgress.stage : compilationProgress.stage || 'Preparando clipes selecionados'}</p><div className="progress-track"><span style={{ width: `${compiledDownloadPercent}%` }} /></div><div className="progress-modal-meta"><b>{compiledDownloadPercent.toFixed(0)}%</b>{compilationProgress.phase === 'download' && compilationProgress.totalBytes > 0 && <small>{(compilationProgress.receivedBytes / 1024 / 1024).toFixed(1)} / {(compilationProgress.totalBytes / 1024 / 1024).toFixed(1)} MB</small>}</div><button onClick={cancelCompilation}>Cancelar</button></section></div>}
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
