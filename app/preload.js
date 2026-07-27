const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gmicro', {
  history: {
    list: () => ipcRenderer.invoke('history:list'),
    load: (id) => ipcRenderer.invoke('history:load', id),
    save: (conv) => ipcRenderer.invoke('history:save', conv),
    delete: (id) => ipcRenderer.invoke('history:delete', id),
    rename: (id, title) => ipcRenderer.invoke('history:rename', id, title),
  },
  onShortcut: (name, cb) => ipcRenderer.on(`shortcut:${name}`, cb),
});
