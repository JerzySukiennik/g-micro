/**
 * Narew Labs — Electron main process.
 *
 * Owns the whole lifecycle of the Python backend, per UI-SPEC.md
 * "Powłoka i architektura": launch spawns it, quit kills it, and a PID-file
 * watchdog cleans up any orphan left behind by a hard crash of Electron
 * itself. The hard requirement from Jurek was explicit: zero processes left
 * running after the window closes, and no terminal ever shown.
 */

const { app, BrowserWindow, Menu, clipboard, dialog, ipcMain, shell } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { spawn, execFileSync } = require('node:child_process');

// The display name is Narew Labs; the on-disk identity stays "g-micro".
//
// userData is derived from the app name, so renaming the app would silently
// point every path below (conversations, PID files, the bridge preference) at
// a fresh empty folder and strand the history already saved under the old
// name. Pinning userData to the original id keeps that data put through this
// rebrand and any future one. Both calls must happen before the first
// getPath('userData') a few lines down.
app.setName('Narew Labs');
app.setPath('userData', path.join(app.getPath('appData'), 'g-micro'));

const REPO_ROOT = path.join(__dirname, '..');
const VENV_PYTHON = path.join(REPO_ROOT, '.venv', 'bin', 'python');
const SERVER_SCRIPT = path.join(REPO_ROOT, 'runtime', 'server.py');
const BRIDGE_SCRIPT = path.join(REPO_ROOT, 'runtime', 'bridge.py');
const PID_FILE = path.join(app.getPath('userData'), 'backend.pid');
const BRIDGE_PID_FILE = path.join(app.getPath('userData'), 'bridge.pid');
const BRIDGE_PREF = path.join(app.getPath('userData'), 'bridge.json');
const CONV_DIR = path.join(app.getPath('userData'), 'conversations');

let win = null;
let backend = null;
let bridge = null;

// ---------------------------------------------------------------- orphan guard --
function killOrphanFromPreviousRun(pidFile = PID_FILE, label = 'backend') {
  if (!fs.existsSync(pidFile)) return;
  const pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim(), 10);
  if (!Number.isInteger(pid)) return;
  try {
    process.kill(pid, 0);          // throws if the process is not running
    process.kill(pid, 'SIGTERM');  // it is — Electron crashed hard last time
    console.log(`[main] killed orphaned ${label} pid ${pid} from a previous crash`);
  } catch {
    // already dead — nothing to do
  } finally {
    fs.rmSync(pidFile, { force: true });
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

// ------------------------------------------------------------------- bridge --
// Lets the phone reach these models from anywhere, by way of a Realtime
// Database both ends connect *out* to — nothing here listens for the internet.
//
// It is a separate process holding its own copy of the model rather than
// sharing the backend's, which costs roughly half a gigabyte. That buys the
// property that matters more: `python runtime/bridge.py` works on its own, so
// the phone keeps working when this app is closed.
//
// Off unless switched on, and that switch is now the only gate there is.
//
// The room stopped being a secret (open mode, see runtime/bridge.py), so
// anyone who loads the site reaches whichever Mac is currently bridging. That
// is workable precisely because it is off by default and Jurek turns it on for
// as long as he is using it — but it only stays true if "off" is what a fresh
// start means. An absent preference file therefore reads as off, not on.
//
// This reverses the auto-on default from earlier the same day; auto-on was
// safe while a paired secret was required, and stopped being so without one.
function bridgeWanted() {
  try { return Boolean(JSON.parse(fs.readFileSync(BRIDGE_PREF, 'utf8')).enabled); }
  catch { return false; }
}

function setBridgeWanted(enabled) {
  fs.writeFileSync(BRIDGE_PREF, JSON.stringify({ enabled }));
}

function spawnBridge() {
  if (bridge) return;
  bridge = spawn(VENV_PYTHON, [BRIDGE_SCRIPT], { cwd: REPO_ROOT });
  fs.writeFileSync(BRIDGE_PID_FILE, String(bridge.pid));
  bridge.stdout.on('data', (d) => process.stdout.write(`[bridge] ${d}`));
  bridge.stderr.on('data', (d) => process.stderr.write(`[bridge] ${d}`));
  bridge.on('exit', (code) => {
    console.log(`[main] bridge exited (${code})`);
    bridge = null;
    fs.rmSync(BRIDGE_PID_FILE, { force: true });
  });
}

function killBridge() {
  if (!bridge || bridge.killed) return;
  bridge.kill('SIGTERM');
  bridge = null;
  fs.rmSync(BRIDGE_PID_FILE, { force: true });
}

function phoneLink() {
  return execFileSync(VENV_PYTHON, [BRIDGE_SCRIPT, '--print-url'],
                      { cwd: REPO_ROOT, encoding: 'utf8' }).trim();
}

function showPhoneLink() {
  let url;
  try {
    url = phoneLink();
  } catch (e) {
    dialog.showMessageBox(win, { type: 'error', message: 'Nie mogę odczytać adresu',
                                 detail: String(e) });
    return;
  }
  dialog.showMessageBox(win, {
    type: 'info',
    message: 'Adres na telefon',
    // Said plainly, because it is the whole security model now: the address is
    // public and the switch is the gate.
    detail: `${url}\n\nAdres jest otwarty — nie ma linku ani hasła. Dopóki `
          + '„Wpuszczaj telefon” jest wyłączone, nikt się tu nie doczeka '
          + 'odpowiedzi. Gdy włączysz, odpowiada każdy, kto wejdzie na tę '
          + 'stronę, więc włączaj na czas używania.',
    buttons: ['Kopiuj adres', 'Zamknij'],
    defaultId: 0,
    cancelId: 1,
  }).then(({ response }) => {
    if (response === 0) clipboard.writeText(url);
  });
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

ipcMain.handle('history:rename', (_e, id, title) => {
  const p = path.join(CONV_DIR, `${id}.json`);
  if (!fs.existsSync(p)) return false;
  const conv = JSON.parse(fs.readFileSync(p, 'utf8'));
  // Titles are otherwise the first 60 characters of the opening message, which
  // stops distinguishing anything once several conversations start alike.
  conv.title = String(title || '').trim().slice(0, 80) || conv.title;
  fs.writeFileSync(p, JSON.stringify(conv));
  return true;
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
        { type: 'separator' },
        // The neurons/probabilities panel was removed on 2026-07-28. What is
        // left here are the two things that genuinely change behaviour: the
        // retrieval toggle, and a way back to the introduction.
        { label: 'Używaj Wikipedii', type: 'checkbox', checked: false,
          click: (item) => win?.webContents.send('shortcut:rag', item.checked) },
        { type: 'separator' },
        { label: 'Wpuszczaj telefon', type: 'checkbox', checked: bridgeWanted(),
          click: (item) => {
            setBridgeWanted(item.checked);
            item.checked ? spawnBridge() : killBridge();
          } },
        { label: 'Adres na telefon…', click: showPhoneLink },
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
  // Two bridges on one room would both answer every job and race each other's
  // writes, so a survivor of a hard crash has to go before a new one starts.
  killOrphanFromPreviousRun(BRIDGE_PID_FILE, 'bridge');
  spawnBackend();
  if (bridgeWanted()) spawnBridge();
  buildMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  killBackend();
  killBridge();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => { killBackend(); killBridge(); });
app.on('will-quit', () => { killBackend(); killBridge(); });
