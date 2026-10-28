// 说 · Shuo — Tauri 3 入口
//
// SPDX-License-Identifier: GPL-3.0-or-later

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod asr;
mod audio;
mod config;
mod denoise;
mod dsp;
mod gguf_asr;
mod hotkey;
mod model_download;
#[cfg(target_os = "linux")]
mod overlay_linux;
mod mel;
mod models;
mod state;
mod typing;

use std::sync::Arc;

use serde::Serialize;
use tauri::{Emitter, Manager};

use config::Config;
use state::{AppState, Status};

/// 追加一行日志到 `~/.shuo/shuo.log`（UTF-8）。
pub fn log_line(msg: &str) {
  let line = format!("{} [INFO] {}\n", config::chrono_time(), msg);
  if let Ok(mut f) = std::fs::OpenOptions::new()
    .create(true)
    .append(true)
    .open(config::log_path())
  {
    use std::io::Write;
    let _ = f.write_all(line.as_bytes());
  }
  eprintln!("shuo: {msg}");
}

/// 确保存在与门户应用 ID 匹配的 `.desktop` 文件。
///
/// xdg-desktop-portal 注册 host app 时会用 `GDesktopAppInfo` 查
/// `com.shuo.desktop.desktop`（且 Exec 必须是真实存在的路径）；deb 已安装
/// 系统级文件，AppImage / 直接运行二进制时补一份到 `~/.local/share/applications`。
/// `NoDisplay=true` 保证它不会在应用菜单里多出一项。
fn ensure_portal_desktop_file() {
  #[cfg(target_os = "linux")]
  {
    if std::path::Path::new("/usr/share/applications/com.shuo.desktop.desktop").is_file() {
      return;
    }
    let exe = std::env::var_os("APPIMAGE")
      .map(std::path::PathBuf::from)
      .or_else(|| std::env::current_exe().ok());
    let Some(exe) = exe else { return };
    let data_home = std::env::var_os("XDG_DATA_HOME")
      .map(std::path::PathBuf::from)
      .or_else(|| {
        std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share"))
      });
    let Some(data_home) = data_home else { return };
    let dir = data_home.join("applications");
    if std::fs::create_dir_all(&dir).is_err() {
      return;
    }
    let exe_s = exe.display().to_string();
    let exec = if exe_s.contains(' ') {
      format!("\"{exe_s}\"")
    } else {
      exe_s
    };
    let content = format!(
      "[Desktop Entry]\nType=Application\nName=Shuo\nComment=Local speech recognition\n\
       Exec={exec}\nTerminal=false\nNoDisplay=true\nStartupWMClass=shuo\n"
    );
    let path = dir.join("com.shuo.desktop.desktop");
    let same = std::fs::read_to_string(&path).map(|old| old == content).unwrap_or(false);
    if !same && std::fs::write(&path, content).is_ok() {
      log_line(&format!("已写入门户桌面文件：{}", path.display()));
    }
  }
}

/// 向 xdg-desktop-portal 注册应用 ID（非沙箱应用必须显式注册）。
///
/// GNOME 的 GlobalShortcuts / RemoteDesktop 会以「An app id is required」拒绝
/// 未注册的进程；注册一次即可（ashpd 内部复用同一条会话总线连接）。
pub fn register_portal_app() {
  #[cfg(target_os = "linux")]
  {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
      let id: ashpd::AppID = match "com.shuo.desktop".parse() {
        Ok(id) => id,
        Err(e) => {
          log_line(&format!("应用 ID 无效：{e}"));
          return;
        }
      };
      match futures_lite::future::block_on(ashpd::register_host_app(id)) {
        Ok(()) => log_line("已向桌面门户注册应用 ID：com.shuo.desktop"),
        Err(e) => log_line(&format!("桌面门户注册失败：{e}")),
      }
    });
  }
}

/// 推理线程数：按核数（上限 16），避免在低核机器上过订阅。
pub fn threads() -> usize {
  std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(4)
    .clamp(1, 16)
}

/// 屏幕底部的胶囊浮层：Linux 走 XWayland helper（不抢焦点），其它平台走独立窗口。
#[derive(Clone, Copy)]
pub enum BusyMode {
  Recording,
  Processing,
}

#[cfg(target_os = "linux")]
pub fn set_busy(_app: &tauri::AppHandle, mode: Option<BusyMode>) {
  let cmd = match mode {
    Some(BusyMode::Recording) => "recording",
    Some(BusyMode::Processing) => "processing",
    None => "idle",
  };
  overlay_linux::send(cmd);
}

#[cfg(not(target_os = "linux"))]
pub fn set_busy(app: &tauri::AppHandle, mode: Option<BusyMode>) {
  let Some(w) = app.get_webview_window("busy") else { return };
  let Some(mode) = mode else {
    let _ = w.hide();
    return;
  };
  let lang = crate::config::load_config().language;
  let zh = lang.starts_with("zh") || lang == "yue";
  let (kind, label) = match mode {
    BusyMode::Recording => ("recording", if zh { "录音中" } else { "Recording" }),
    BusyMode::Processing => ("processing", if zh { "识别中" } else { "Working" }),
  };
  let _ = app.emit_to(
    "busy",
    "busy-state",
    serde_json::json!({ "mode": kind, "label": label }),
  );
  place_busy(&w);
  let _ = w.show();
}

/// 贴在屏幕底部、任务栏上方一点（仅非 Linux 的独立窗口用）。
#[cfg(not(target_os = "linux"))]
fn place_busy(w: &tauri::WebviewWindow) {
  use tauri::Manager;
  let Ok(Some(mon)) = w.primary_monitor() else { return };
  let scale = mon.scale_factor();
  let screen = mon.size();
  let win_w = 300.0 * scale;
  let win_h = 72.0 * scale;
  let x = ((screen.width as f64 - win_w) / 2.0).max(0.0) as i32;
  let y = (screen.height as f64 - win_h - 96.0 * scale).max(0.0) as i32;
  let _ = w.set_position(tauri::PhysicalPosition::new(x, y));
}

pub fn denoise_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
  let mut candidates = Vec::new();
  if let Ok(res) = app.path().resource_dir() {
    candidates.push(res.join("denoise.onnx"));
  }
  candidates.push(
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("resources/denoise.onnx"),
  );
  if let Ok(exe) = std::env::current_exe()
    && let Some(dir) = exe.parent()
  {
    candidates.push(dir.join("denoise.onnx"));
    candidates.push(dir.join("resources/denoise.onnx"));
  }
  candidates.into_iter().find(|p| p.is_file())
}

#[derive(Serialize)]
struct Bootstrap {
  config: Config,
  status: Status,
  history: Vec<config::HistoryItem>,
  platform: String,
  version: String,
  hotkey_supported: bool,
  model_hint: String,
}

#[tauri::command]
fn bootstrap(state: tauri::State<'_, Arc<AppState>>) -> Bootstrap {
  let st = state.inner();
  let cfg = st.config();
  let status = st.status.lock().unwrap().clone();
  let mut history = st.history.lock().unwrap().clone();
  history.reverse();
  let paths = st.paths.lock().unwrap().clone();
  Bootstrap {
    config: cfg,
    status,
    history,
    platform: std::env::consts::OS.to_string(),
    version: env!("CARGO_PKG_VERSION").to_string(),
    hotkey_supported: cfg!(windows) || cfg!(target_os = "linux"),
    model_hint: models::describe(&paths),
  }
}

#[tauri::command]
fn get_status(state: tauri::State<'_, Arc<AppState>>) -> Status {
  state.status.lock().unwrap().clone()
}

/// 合并式保存配置（允许只传变化字段，保留 Python 版写入的其它键）。
#[tauri::command]
fn save_config(
  patch: serde_json::Value,
  state: tauri::State<'_, Arc<AppState>>,
) -> Result<Config, String> {
  let current = state.config();
  let mut base = serde_json::to_value(&current).map_err(|e| e.to_string())?;
  if let (Some(obj), Some(patch_obj)) = (base.as_object_mut(), patch.as_object()) {
    for (k, v) in patch_obj {
      obj.insert(k.clone(), v.clone());
    }
  }
  let next: Config = serde_json::from_value(base).map_err(|e| format!("配置无效：{e}"))?;
  config::save_config(&next)?;
  *state.cfg.lock().unwrap() = next.clone();

  // 麦克风 / 模型 / 热键变化后按需重启组件
  let mic_changed = current.mic_device != next.mic_device;
  let hotkey_changed = current.hotkey != next.hotkey
    || current.hotkey_terminal != next.hotkey_terminal
    || current.hotkey_enabled != next.hotkey_enabled
    || current.hotkey_terminal_enabled != next.hotkey_terminal_enabled;
  if mic_changed {
    if let Err(e) = state.inner().restart_recorder() {
      return Err(e);
    }
  }
  if hotkey_changed {
    state.inner().start_hotkey();
  }
  if current.asr_model_dir != next.asr_model_dir || current.asr_backend != next.asr_backend {
    state.inner().reload_models();
  }
  Ok(next)
}

#[tauri::command]
fn set_hotkey(
  spec: String,
  which: Option<String>,
  state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
  if hotkey::parse_spec(&spec).is_none() {
    return Err(format!("无法识别的热键：{spec}"));
  }
  let mut cfg = state.config();
  if which.as_deref() == Some("terminal") {
    cfg.hotkey_terminal = spec;
  } else {
    cfg.hotkey = spec;
  }
  config::save_config(&cfg)?;
  *state.cfg.lock().unwrap() = cfg;
  state.inner().start_hotkey();
  Ok(())
}

#[tauri::command]
fn list_mics() -> Vec<(String, bool)> {
  audio::list_input_devices()
}

#[tauri::command]
fn start_record(state: tauri::State<'_, Arc<AppState>>) {
  state.inner().press(crate::typing::PasteMode::CtrlV);
}

#[tauri::command]
fn stop_record(state: tauri::State<'_, Arc<AppState>>) {
  state.inner().release();
}

#[tauri::command]
fn copy_text(text: String) -> Result<(), String> {
  typing::copy_text(&text)
}

#[tauri::command]
fn get_history(state: tauri::State<'_, Arc<AppState>>) -> Vec<config::HistoryItem> {
  let mut items = state.history.lock().unwrap().clone();
  items.reverse();
  items
}

#[tauri::command]
fn clear_history(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
  config::save_history(&[])?;
  state.history.lock().unwrap().clear();
  Ok(())
}

/// 输出风格提示词预设（进 ASR chat 模板 system 段，与微调模型包 shuo-model.json 记录的提示词逐字一致）
#[tauri::command]
fn list_style_presets() -> Vec<PresetInfo> {
  asr::style_preset_list()
    .into_iter()
    .map(|(key, zh, en)| PresetInfo { key, name_zh: zh, name_en: en })
    .collect()
}

#[tauri::command]
fn get_style_preset_prompt(key: String) -> String {
  asr::style_preset_prompt(&key).unwrap_or("").to_string()
}

#[derive(Serialize)]
struct PresetInfo {
  key: &'static str,
  name_zh: &'static str,
  name_en: &'static str,
}

#[tauri::command]
fn restart_recorder(state: tauri::State<'_, Arc<AppState>>) -> Result<(), String> {
  state.inner().restart_recorder()
}

#[tauri::command]
fn reload_engine(state: tauri::State<'_, Arc<AppState>>) {
  state.inner().reload_engine_full();
}

/// 一键下载微调模型包（GitHub Release → ~/.shuo/models/，后台线程 + 进度事件）
#[tauri::command]
fn download_model(
  app: tauri::AppHandle,
  state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
  model_download::start(app, state.inner().clone())
}

#[derive(Serialize)]
struct DownloadFile {
  name: String,
  present: bool,
  size: u64,
}

#[derive(Serialize)]
struct DownloadInfo {
  dir: String,
  url: String,
  downloading: bool,
  files: Vec<DownloadFile>,
}

/// 下载源与本地文件状态（前端初始化/恢复按钮态用）
#[tauri::command]
fn model_download_info() -> DownloadInfo {
  DownloadInfo {
    dir: model_download::target_dir().display().to_string(),
    url: model_download::release_url(),
    downloading: model_download::is_downloading(),
    files: model_download::local_state()
      .into_iter()
      .map(|(name, present, size)| DownloadFile { name, present, size })
      .collect(),
  }
}

/// 在系统浏览器打开下载源 Release 页
#[tauri::command]
fn open_release_page() {
  open_path(std::path::Path::new(&model_download::release_url()));
}

#[tauri::command]
fn open_user_dir() {
  let dir = config::user_dir();
  open_path(&dir);
}

#[tauri::command]
fn open_log() {
  open_path(&config::log_path());
}

fn open_path(path: &std::path::Path) {
  #[cfg(windows)]
  let _ = std::process::Command::new("explorer").arg(path).spawn();
  #[cfg(target_os = "linux")]
  let _ = std::process::Command::new("xdg-open").arg(path).spawn();
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
  app.exit(0);
}

#[tauri::command]
fn hide_window(app: tauri::AppHandle) {
  if let Some(w) = app.get_webview_window("main") {
    let _ = w.hide();
  }
}

fn main() {
  let _ = config::user_dir();

  tauri::Builder::default()
    .runtime(tauri_runtime_wry::Wry::default())
    .setup(|app| {
      // 打包资源目录里的降噪模型（deb: /usr/lib/Shuo/denoise.onnx）
      let _ = app.path().resource_dir();

      let cfg = config::load_config();
      log_line(&format!("启动 Shuo {}（{}）", env!("CARGO_PKG_VERSION"), std::env::consts::OS));
      let state = AppState::new(app.handle().clone(), cfg);
      app.manage(state.clone());

      ensure_portal_desktop_file();
      register_portal_app();
      #[cfg(target_os = "linux")]
      overlay_linux::start(app.handle());

      if let Err(e) = state.restart_recorder() {
        log_line(&format!("录音初始化失败：{e}"));
      }
      state.start_hotkey();

      // 托盘失败不致命：GNOME 没有 StatusNotifierWatcher 时 ksni 直接报错
      if let Err(e) = setup_tray(app) {
        log_line(&format!("托盘不可用（{e}）——窗口与识别照常工作"));
      }

      if state.config().start_hidden
        && let Some(w) = app.get_webview_window("main")
      {
        let _ = w.hide();
      }
      Ok(())
    })
    .on_window_event(|window, event| match event {
      tauri::WindowEvent::CloseRequested { api, .. } => {
        api.prevent_close();
        let _ = window.hide();
      }
      tauri::WindowEvent::Resized(_) if window.is_minimized().unwrap_or(false) => {
        let _ = window.hide();
      }
      _ => {}
    })
    .invoke_handler(tauri::generate_handler![
      bootstrap,
      get_status,
      save_config,
      set_hotkey,
      list_mics,
      start_record,
      stop_record,
      copy_text,
      get_history,
      clear_history,
      list_style_presets,
      get_style_preset_prompt,
      restart_recorder,
      reload_engine,
      download_model,
      model_download_info,
      open_release_page,
      open_user_dir,
      open_log,
      quit_app,
      hide_window
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
  use tauri::menu::{Menu, MenuItem};
  use tauri::tray::TrayIconBuilder;

  let show = MenuItem::with_id(app, "show", "显示 / 隐藏窗口", true, None::<&str>)?;
  let quit = MenuItem::with_id(app, "quit", "退出 Shuo", true, None::<&str>)?;
  let menu = Menu::with_items(app, &[&show, &quit])?;
  let app_ev = app.handle().clone();
  let app_menu = app.handle().clone();

  let mut builder = TrayIconBuilder::with_id("main")
    .tooltip("说 · Shuo")
    .menu(&menu)
    .show_menu_on_left_click(false)
    .on_menu_event(move |_app, event| match event.id.as_ref() {
      "show" => toggle_window(&app_menu),
      "quit" => _app.exit(0),
      _ => {}
    })
    .on_tray_icon_event(move |_tray, event| {
      if let tauri::tray::TrayIconEvent::Click {
        button: tauri::tray::MouseButton::Left,
        button_state: tauri::tray::MouseButtonState::Up,
        ..
      } = event
      {
        toggle_window(&app_ev);
      }
    });
  if let Ok(icon) = tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png")) {
    builder = builder.icon(icon);
  }
  builder.build(app)?;
  Ok(())
}

fn toggle_window(app: &tauri::AppHandle) {
  if let Some(w) = app.get_webview_window("main") {
    if w.is_visible().unwrap_or(true) {
      let _ = w.hide();
    } else {
      let _ = w.unminimize();
      let _ = w.show();
      let _ = w.set_focus();
    }
  }
}

#[allow(dead_code)]
fn announce(app: &tauri::AppHandle, msg: &str) {
  let _ = app.emit("notice", msg);
}
