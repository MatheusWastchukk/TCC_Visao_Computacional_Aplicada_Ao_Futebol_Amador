const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('processing', {
  start: (mode) => ipcRenderer.invoke('processing-start', mode),
});
