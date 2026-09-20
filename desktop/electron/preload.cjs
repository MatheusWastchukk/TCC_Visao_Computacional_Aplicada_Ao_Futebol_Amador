const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('desktop', {
  selectVideo: () => ipcRenderer.invoke('select-video'),
  download: (url) => ipcRenderer.invoke('download-url', url),
});
