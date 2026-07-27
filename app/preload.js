const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gmicro', {
  history: {
    list: () => ipcRenderer.invoke('history:list'),
    load: (id) => ipcRenderer.invoke('history:load', id),
    save: (conv) => ipcRenderer.invoke('history:save', conv),
    delete: (id) => ipcRenderer.invoke('history:delete', id),
  },
  onShortcut: (name, cb) => ipcRenderer.on(`shortcut:${name}`, cb),
});
