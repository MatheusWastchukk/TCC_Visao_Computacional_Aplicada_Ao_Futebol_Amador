const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('desktop', {
  selectVideo: () => ipcRenderer.invoke('select-video'),
  download: (url, downloadId = `${Date.now()}-${Math.random().toString(16).slice(2)}`) => ipcRenderer.invoke('download-url', { url, downloadId }),
  cancelDownload: (downloadId) => ipcRenderer.invoke('cancel-download', downloadId),
  onDownloadProgress: (callback) => {
    const listener = (_event, progress) => callback(progress);
    ipcRenderer.on('download-progress', listener);
    return () => ipcRenderer.removeListener('download-progress', listener);
  },
});
