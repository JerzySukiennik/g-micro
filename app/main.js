/**
 * G-Micro — Electron main process.
 *
 * Owns the whole lifecycle of the Python backend, per UI-SPEC.md
 * "Powłoka i architektura": launch spawns it, quit kills it, and a PID-file
 * watchdog cleans up any orphan left behind by a hard crash of Electron
 * itself. The hard requirement from Jurek was explicit: zero processes left
 * running after the window closes, and no terminal ever shown.
 */

const { app, BrowserWindow, Menu, ipcMain, shell } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const { spawn } = require('node:child_process');

const REPO_ROOT = path.join(__dirname, '..');
const VENV_PYTHON = path.join(REPO_ROOT, '.venv', 'bin', 'python');
const SERVER_SCRIPT = path.join(REPO_ROOT, 'runtime', 'server.py');
const PID_FILE = path.join(app.getPath('userData'), 'backend.pid');
const CONV_DIR = path.join(app.getPath('userData'), 'conversations');

let win = null;
let backend = null;

// ---------------------------------------------------------------- orphan guard --
function killOrphanFromPreviousRun() {
  if (!fs.existsSync(PID_FILE)) return;
  const pid = parseInt(fs.readFileSync(PID_FILE, 'utf8').trim(), 10);
  if (!Number.isInteger(pid)) return;
  try {
    process.kill(pid, 0);          // throws if the process is not running
    process.kill(pid, 'SIGTERM');  // it is — Electron crashed hard last time
    console.log(`[main] killed orphaned backend pid ${pid} from a previous crash`);
  } catch {
    // already dead — nothing to do
  } finally {
    fs.rmSync(PID_FILE, { force: true });
  }
}

function spawnBackend() {
  backend = spawn(VENV_PYTHON, [SERVER_SCRIPT], { cwd: REPO_ROOT });
  fs.writeFileSync(PID_FILE, String(backend.pid));
  backend.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  backend.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  backend.on('exit', (code) => {
    console.log(`[main] backend exited (${code})`);
    fs.rmSync(PID_FILE, { force: true });
  });
}

function killBackend() {
  if (!backend || backend.killed) return;
  backend.kill('SIGTERM');
  fs.rmSync(PID_FILE, { force: true });
}

// -------------------------------------------------------------------- window --
function createWindow() {
  win = new BrowserWindow({
    width: 1180, height: 760, minWidth: 820, minHeight: 560,
    backgroundColor: '#000000',
    titleBarStyle: 'hiddenInset',   // native traffic lights, no title text —
    trafficLightPosition: { x: 16, y: 13 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  // Renderer failures are invisible from the terminal otherwise, which cost a
  // debugging session: the backend logged "listening", no window appeared, and
  // there was nothing anywhere saying why. Forwarding the renderer's console
  // and its crash events into the main log makes `npm start` self-diagnosing.
  win.webContents.on('console-message', (_e, level, message, line, source) => {
    if (level >= 2) console.log(`[renderer] ${message}  (${source}:${line})`);
  });
  win.webContents.on('did-fail-load', (_e, code, desc, url) => {
    console.log(`[renderer] did-fail-load ${code} ${desc} ${url}`);
  });
  win.webContents.on('render-process-gone', (_e, details) => {
    console.log(`[renderer] process gone: ${JSON.stringify(details)}`);
  });
  win.on('unresponsive', () => console.log('[renderer] unresponsive'));

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  win.on('closed', () => { win = null; });
}

// -------------------------------------------------------------- conversations --
function ensureConvDir() { fs.mkdirSync(CONV_DIR, { recursive: true }); }

ipcMain.handle('history:list', () => {
  ensureConvDir();
  return fs.readdirSync(CONV_DIR)
    .filter((f) => f.endsWith('.json'))
    .map((f) => {
      const p = path.join(CONV_DIR, f);
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      return { id: f.replace(/\.json$/, ''), title: data.title, updated: data.updated };
    })
    .sort((a, b) => b.updated - a.updated);
});

ipcMain.handle('history:load', (_e, id) => {
  const p = path.join(CONV_DIR, `${id}.json`);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
});

ipcMain.handle('history:save', (_e, conv) => {
  ensureConvDir();
  const id = conv.id || String(Date.now());
  fs.writeFileSync(path.join(CONV_DIR, `${id}.json`), JSON.stringify({ ...conv, id }));
  return id;
});

ipcMain.handle('history:delete', (_e, id) => {
  fs.rmSync(path.join(CONV_DIR, `${id}.json`), { force: true });
});

// ---------------------------------------------------------------------- menu --
function buildMenu() {
  const template = [
    {
      label: app.name,
      submenu: [
        { role: 'about' }, { type: 'separator' },
        { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' },
        { type: 'separator' }, { role: 'quit' },
      ],
    },
    {
      label: 'File',
      submenu: [
        { label: 'New Conversation', accelerator: 'CmdOrCtrl+N',
          click: () => win?.webContents.send('shortcut:new') },
        { label: 'Toggle History', accelerator: 'CmdOrCtrl+K',
          click: () => win?.webContents.send('shortcut:history') },
        // No settings item: sampling is fixed at measured defaults and there
        // is nothing left for a user to configure. This toggles the neurons
        // and probabilities panel instead — hidden by default so the app
        // reads as a chat, still one shortcut away.
        { label: 'Pokaż podgląd modelu', accelerator: 'CmdOrCtrl+Alt+D',
          click: () => win?.webContents.send('shortcut:panel') },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' }, { role: 'redo' }, { type: 'separator' },
        { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' },
      ],
    },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// -------------------------------------------------------------------- lifecycle --
app.whenReady().then(() => {
  killOrphanFromPreviousRun();
  spawnBackend();
  buildMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  killBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', killBackend);
app.on('will-quit', killBackend);
