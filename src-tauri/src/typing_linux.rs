// 说 · Shuo — Linux 自动输入（上屏）
//
// 为什么不是逐字符 keysym 注入：GNOME 的 RemoteDesktop 门户把 keysym 交给合成器后，
// **当前输入法（libpinyin）会把它当键盘输入处理** —— 实测注入 `abc123` 上屏成
// 「阿拔出23」。中文更糟（keysym 被拼音候选吃掉）。
//
// 可靠通路（按顺序）：
//   1. 剪贴板 + 粘贴快捷键（Ctrl+V / Ctrl+Shift+V，GTK/Qt/终端/Chromium 都认）：
//      文本不经过输入法，中文/英文/符号都能原样上屏；先把原剪贴板存起来，粘完再还原。
//   2. RemoteDesktop 门户（只用来发快捷键，不发正文）——**后台申请**，绝不阻塞上屏。
//   3. wtype / ydotool / xdotool 发同样的快捷键（X11 / wlroots / uinput 环境）。
//   4. 都失败：只把文本放进剪贴板，并明确告诉用户手动粘贴。
//
// ⚠️ 剪贴板**不再优先用 wl-clipboard**：
//   wl-copy / wl-paste 为了拿到选区必须临时映射一个透明窗口（GNOME 里显示成
//   「wl-clipboard」窗口）。Ubuntu 26.04 自带的 wl-clipboard 2.2.x 在 GNOME 上还会
//   触发焦点保护：窗口没被点一下，剪贴板就不生效 —— 连续说话时表现为「弹一个
//   wl-clipboard 的框，不点它后面所有结果都出不来」。
//   改用 XWayland 的 X 选区（xclip / xsel，mutter 会把 X11 ↔ Wayland 剪贴板双向桥接）：
//   没有窗口、不抢焦点、不会弹框。只有纯 Wayland（无 XWayland）时才退回 wl-clipboard。
//
// ⚠️ 所有外部命令都有硬超时，且整条上屏链路跑在独立线程（state.rs 的 shuo-output）：
//   任何一个工具卡住都只会丢掉这一条文本，绝不会拖住录音和识别。
//
// 注意：真正的「输入法通路」（IBus 引擎 commit_text）可以彻底避开模拟按键，但需要
// 用户切到「Shuo Voice」输入法（或应用临时切换引擎再切回），代价见
// docs/scaffold-comparison.md 的讨论；默认不启用。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ashpd::desktop::{PersistMode, Session};
use ashpd::desktop::remote_desktop::{
  DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions, StartOptions,
};
use enumflags2::BitFlags;

use crate::typing::PasteMode;

#[path = "uinput.rs"]
mod uinput;

/// 单个外部命令（剪贴板工具 / 注入工具）的硬超时。
/// 卡住的工具杀掉即可，后面还有别的通道；绝不允许无期限等待。
const CMD_TIMEOUT: Duration = Duration::from_millis(1500);
/// 门户发快捷键的超时（D-Bus 卡住时同理）。
const PORTAL_TIMEOUT: Duration = Duration::from_millis(1500);
/// 授权失败后的冷却时间，避免反复弹框。
const PORTAL_COOLDOWN: Duration = Duration::from_secs(30);

struct PortalText {
  rd: RemoteDesktop,
  session: Session<RemoteDesktop>,
}

/// 门户（RemoteDesktop）的授权状态机。
///
/// 关键点：申请授权可能在用户点「共享」之前一直挂着，所以**不能在调用线程里等**，
/// 只能在后台线程里申请；上屏线程看到「还没授权」就直接走别的通道。
enum PortalState {
  Idle,
  /// 后台正在等用户确认
  Pending,
  Ready(PortalText),
  /// 申请失败/被拒绝，冷却到该时刻再试
  Cooldown(Instant),
}

static PORTAL: OnceLock<Mutex<PortalState>> = OnceLock::new();

/// 剪贴板写入代数：用来避免「延迟还原剪贴板」把下一次上屏的内容覆盖掉。
static CLIP_GEN: AtomicU64 = AtomicU64::new(0);

// evdev keycode（门户的 NotifyKeyboardKeycode 用这个，能正确带修饰键；
// keysym 只适合字符，带 Ctrl/Shift 的组合会被输入法/合成器吃掉）
const KEY_LEFTCTRL: i32 = 29;
const KEY_LEFTSHIFT: i32 = 42;
const KEY_V: i32 = 47;
const KEY_INSERT: i32 = 110;

fn has(cmd: &str) -> bool {
  // 直接查 PATH，不 spawn `sh -c`（上屏链路上每一次多余进程都会增加卡住的机会）
  let Some(path) = std::env::var_os("PATH") else {
    return false;
  };
  std::env::split_paths(&path).any(|dir| {
    let p = dir.join(cmd);
    p.is_file()
  })
}

fn has_x11() -> bool {
  std::env::var_os("DISPLAY").is_some_and(|v| !v.is_empty())
}

fn has_wayland() -> bool {
  std::env::var_os("WAYLAND_DISPLAY").is_some_and(|v| !v.is_empty())
}

/// 等子进程退出，超时直接杀掉。返回退出码是否成功。
fn wait_timeout(child: &mut std::process::Child, cmd: &str, timeout: Duration) -> Result<(), String> {
  let deadline = Instant::now() + timeout;
  loop {
    match child.try_wait() {
      Ok(Some(st)) => {
        return if st.success() {
          Ok(())
        } else {
          Err(format!("{cmd} 退出码 {:?}", st.code()))
        };
      }
      Ok(None) => {}
      Err(e) => return Err(format!("{cmd} 等待失败：{e}")),
    }
    if Instant::now() >= deadline {
      let _ = child.kill();
      let _ = child.wait();
      return Err(format!("{cmd} 超时（{} ms）", timeout.as_millis()));
    }
    std::thread::sleep(Duration::from_millis(10));
  }
}

/// 跑一个「只关心成功与否」的命令（stdout/stderr 丢弃，可喂 stdin）；带硬超时。
///
/// stdout/stderr 用 null 而不是管道：xclip/wl-copy 这类工具会 fork 一个常驻进程持有
/// 选区，若继承管道会让读端永远等不到 EOF —— 这正是很多「上屏卡死」的来源。
fn run_quiet(
  cmd: &str,
  args: &[&str],
  stdin_text: Option<&str>,
  timeout: Duration,
) -> Result<(), String> {
  let mut c = Command::new(cmd);
  c.args(args)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .stdin(if stdin_text.is_some() { Stdio::piped() } else { Stdio::null() });
  let mut child = c.spawn().map_err(|e| format!("启动 {cmd} 失败：{e}"))?;
  if let Some(text) = stdin_text
    && let Some(mut sin) = child.stdin.take()
  {
    // 子进程（或其 fork 出的常驻进程）可能提前关闭管道：写入失败不算失败
    let _ = sin.write_all(text.as_bytes());
  }
  wait_timeout(&mut child, cmd, timeout)
}

/// 跑一个需要读 stdout 的命令；带硬超时，读线程不会拖住调用方。
fn run_capture(cmd: &str, args: &[&str], timeout: Duration) -> Option<(bool, Vec<u8>)> {
  let mut child = Command::new(cmd)
    .args(args)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::null())
    .spawn()
    .ok()?;
  let mut out = child.stdout.take()?;
  let (tx, rx) = std::sync::mpsc::channel();
  std::thread::spawn(move || {
    let mut buf = Vec::new();
    let _ = out.read_to_end(&mut buf);
    let _ = tx.send(buf);
  });
  let deadline = Instant::now() + timeout;
  let ok = loop {
    match child.try_wait() {
      Ok(Some(st)) => break st.success(),
      Ok(None) => {}
      Err(_) => break false,
    }
    if Instant::now() >= deadline {
      let _ = child.kill();
      let _ = child.wait();
      return None;
    }
    std::thread::sleep(Duration::from_millis(10));
  };
  // fork 出的常驻进程可能还握着管道：等一小会儿拿不到就当空
  let buf = rx.recv_timeout(Duration::from_millis(300)).unwrap_or_default();
  Some((ok, buf))
}

/// 从剪贴板原始字节解出可还原的文本。
///
/// 剪贴板里可能是图片/文件列表等二进制内容：这类一律当作「读不出来」，
/// 免得以乱码文本回写覆盖用户原本的剪贴板。
fn clipboard_bytes_to_text(bytes: Vec<u8>) -> Option<String> {
  const MAX_RESTORE: usize = 1 << 20; // 1 MiB：再大就不折腾还原了
  if bytes.is_empty() || bytes.len() > MAX_RESTORE || bytes.contains(&0) {
    return None;
  }
  String::from_utf8(bytes).ok()
}

pub fn type_text(text: &str, mode: PasteMode) -> Result<(), String> {
  if text.is_empty() {
    return Ok(());
  }

  // 1. 剪贴板 + 粘贴快捷键
  match clipboard_paste(text, mode) {
    Ok(()) => return Ok(()),
    Err(e) => crate::log_line(&format!("剪贴板上屏失败：{e}")),
  }

  // 2. 外部工具直接打字（ASCII 环境 / 没有剪贴板工具时）
  if has("wtype") && run_quiet("wtype", &["-"], Some(text), CMD_TIMEOUT).is_ok() {
    return Ok(());
  }
  if has("ydotool")
    && run_quiet("ydotool", &["type", "--file", "-"], Some(text), CMD_TIMEOUT).is_ok()
  {
    return Ok(());
  }
  if has("xdotool")
    && run_quiet(
      "xdotool",
      &["type", "--clearmodifiers", "--file", "-"],
      Some(text),
      CMD_TIMEOUT,
    )
    .is_ok()
  {
    return Ok(());
  }

  // 3. 只放剪贴板
  copy_text(text)?;
  Err("没有可用的上屏通道，已把文本复制到剪贴板，请按 Ctrl+V / Shift+Insert 粘贴".into())
}

fn clipboard_paste(text: &str, mode: PasteMode) -> Result<(), String> {
  let previous = read_clipboard();
  copy_text(text)?;
  // copy_text 内部自增了代数；还原时若代数已变（又粘贴了新内容 / 用户手动复制），就别覆盖
  let generation = CLIP_GEN.load(Ordering::SeqCst);

  match send_paste_shortcut(mode) {
    Ok(()) => {
      // 目标应用可能异步读剪贴板，延迟后再还原
      if let Some(prev) = previous {
        std::thread::spawn(move || {
          std::thread::sleep(Duration::from_millis(800));
          if CLIP_GEN.load(Ordering::SeqCst) != generation {
            return;
          }
          let _ = copy_text(&prev);
        });
      }
      Ok(())
    }
    Err(e) => {
      if let Some(prev) = previous {
        let _ = copy_text(&prev);
      }
      Err(e)
    }
  }
}

fn uinput_paste(mode: PasteMode) -> Result<(), String> {
  use std::sync::Mutex;
  use std::sync::OnceLock;
  static VK: OnceLock<Mutex<Option<uinput::VirtualKeyboard>>> = OnceLock::new();
  let cell = VK.get_or_init(|| Mutex::new(None));
  let mut guard = cell.lock().map_err(|_| "uinput 状态锁损坏".to_string())?;
  if guard.is_none() {
    *guard = Some(uinput::VirtualKeyboard::open()?);
    crate::log_line("uinput 虚拟键盘已就绪");
  }
  let codes: Vec<u16> = paste_keys(mode).iter().map(|c| *c as u16).collect();
  guard.as_ref().unwrap().tap_combo(&codes)
}

/// 发 Ctrl+V（GUI 应用通用；终端请用 Ctrl+Shift+V，见文档）。
fn send_paste_shortcut(mode: PasteMode) -> Result<(), String> {
  // 首选 uinput：不需要门户授权，也不会每次启动弹「共享」
  if uinput_paste(mode).is_ok() {
    return Ok(());
  }

  // 门户只发快捷键；没授权就去后台申请，本次直接走后面的通道（绝不等对话框）
  match portal_paste(mode) {
    Ok(()) => return Ok(()),
    Err(e) => crate::log_line(&format!("门户粘贴快捷键不可用：{e}")),
  }

  // X11 / wlroots / uinput
  let keys = paste_keys(mode);
  if has("xdotool")
    && run_quiet(
      "xdotool",
      &["key", "--clearmodifiers", paste_combo_name(&keys)],
      None,
      CMD_TIMEOUT,
    )
    .is_ok()
  {
    return Ok(());
  }
  if has("ydotool") {
    let seq: Vec<String> = keys
      .iter()
      .map(|c| format!("{c}:1"))
      .chain(keys.iter().rev().map(|c| format!("{c}:0")))
      .collect();
    let mut args: Vec<&str> = vec!["key"];
    args.extend(seq.iter().map(String::as_str));
    if run_quiet("ydotool", &args, None, CMD_TIMEOUT).is_ok() {
      return Ok(());
    }
  }
  Err("没有可用的粘贴快捷键通道（uinput / 门户 / xdotool 都不可用）".into())
}

/// 上屏快捷键：主热键 → Ctrl+V（图形界面）；终端热键 → Ctrl+Shift+V（终端/TUI）。
fn paste_keys(mode: PasteMode) -> Vec<i32> {
  match mode {
    PasteMode::CtrlV => vec![KEY_LEFTCTRL, KEY_V],
    PasteMode::CtrlShiftV => vec![KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_V],
  }
}

fn paste_combo_name(keys: &[i32]) -> &'static str {
  match keys {
    [KEY_LEFTCTRL, KEY_V] => "ctrl+v",
    [KEY_LEFTSHIFT, KEY_INSERT] => "shift+Insert",
    _ => "ctrl+shift+v",
  }
}

fn portal_cell() -> &'static Mutex<PortalState> {
  PORTAL.get_or_init(|| Mutex::new(PortalState::Idle))
}

/// 尽力用门户发一次快捷键；**任何情况下都不等授权对话框**。
fn portal_paste(mode: PasteMode) -> Result<(), String> {
  let cell = portal_cell();
  let mut st = cell.lock().map_err(|_| "门户状态锁已损坏".to_string())?;
  match &*st {
    PortalState::Ready(p) => {
      let keys = paste_keys(mode);
      let result = futures_lite::future::block_on(async {
        let send = async {
          for &code in &keys {
            p.rd
              .notify_keyboard_keycode(&p.session, code, KeyState::Pressed, Default::default())
              .await
              .map_err(|e| format!("{e}"))?;
            async_io::Timer::after(Duration::from_millis(12)).await;
          }
          for &code in keys.iter().rev() {
            p.rd
              .notify_keyboard_keycode(&p.session, code, KeyState::Released, Default::default())
              .await
              .map_err(|e| format!("{e}"))?;
            async_io::Timer::after(Duration::from_millis(12)).await;
          }
          Ok::<(), String>(())
        };
        match futures_lite::future::or(
          async { Some(send.await) },
          async {
            async_io::Timer::after(PORTAL_TIMEOUT).await;
            None
          },
        )
        .await
        {
          Some(r) => r,
          None => Err("门户发送快捷键超时".into()),
        }
      });
      if result.is_err() {
        // 会话被撤回/失效：丢掉连接和 token，下次上屏再后台申请一次
        *st = PortalState::Idle;
        let _ = std::fs::remove_file(token_path());
      }
      result
    }
    PortalState::Pending => Err("门户授权等待中".into()),
    PortalState::Cooldown(t) if Instant::now() < *t => Err("门户授权冷却中".into()),
    PortalState::Cooldown(_) | PortalState::Idle => {
      *st = PortalState::Pending;
      drop(st);
      request_portal_async();
      Err("门户授权已改为后台申请".into())
    }
  }
}

/// 在后台线程里申请 RemoteDesktop 授权（GNOME 会弹一次「共享屏幕/键盘」对话框）。
/// 用户点与不点都不影响当前这次上屏；点完之后后续上屏就能用门户发快捷键。
fn request_portal_async() {
  crate::log_line("后台申请 RemoteDesktop 授权（仅用于粘贴快捷键，不影响当前上屏）");
  if std::thread::Builder::new()
    .name("shuo-portal-auth".into())
    .spawn(|| {
      let token = std::fs::read_to_string(token_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
      // 先用记住的 token；失效则清掉并重新申请（GNOME 会再弹一次框）
      let first = futures_lite::future::block_on(connect_portal(token.as_deref()));
      let result = match first {
        Ok(p) => Ok(p),
        Err(e) if token.is_some() => {
          crate::log_line(&format!("门户授权已失效（{e}），重新申请"));
          let _ = std::fs::remove_file(token_path());
          futures_lite::future::block_on(connect_portal(None))
        }
        Err(e) => Err(e),
      };
      let mut st = portal_cell().lock().unwrap();
      match result {
        Ok(p) => {
          *st = PortalState::Ready(p);
          crate::log_line("RemoteDesktop 键盘注入已授权（后续上屏可用）");
        }
        Err(e) => {
          *st = PortalState::Cooldown(Instant::now() + PORTAL_COOLDOWN);
          crate::log_line(&format!("RemoteDesktop 授权失败：{e}（{}s 后再试）", PORTAL_COOLDOWN.as_secs()));
        }
      }
    })
    .is_err()
  {
    *portal_cell().lock().unwrap() = PortalState::Idle;
  }
}

fn token_path() -> std::path::PathBuf {
  crate::config::user_dir().join("portal_restore_token")
}

async fn connect_portal(token: Option<&str>) -> Result<PortalText, String> {
  let rd = RemoteDesktop::new().await.map_err(|e| format!("{e}"))?;
  let session = rd
    .create_session(Default::default())
    .await
    .map_err(|e| format!("创建会话失败：{e}"))?;
  // 持久化授权：把上次的 restore token 带上，GNOME 就不再每次弹「共享」对话框
  rd.select_devices(
    &session,
    SelectDevicesOptions::default()
      .set_devices(BitFlags::from_flag(DeviceType::Keyboard))
      .set_persist_mode(PersistMode::ExplicitlyRevoked)
      .set_restore_token(token),
  )
  .await
  .map_err(|e| format!("选择键盘设备失败：{e}"))?;
  let start = rd
    .start(&session, None, StartOptions::default())
    .await
    .map_err(|e| format!("请求授权失败（{e}）"))?;
  let devices = start.response().map_err(|e| format!("授权结果无效：{e}"))?;
  if !devices.devices().contains(DeviceType::Keyboard) {
    return Err("用户未授权键盘控制".into());
  }
  if let Some(t) = devices.restore_token() {
    let _ = std::fs::write(token_path(), t);
    crate::log_line("RemoteDesktop 授权已记住（restore token）");
  }
  Ok(PortalText { rd, session })
}

/// 读当前剪贴板（用于粘贴后还原）。
///
/// 优先 X11 选区（xclip/xsel，无窗口）；纯 Wayland 才退 wl-paste —— wl-paste 同样会
/// 临时映射窗口，正是「wl-clipboard 弹框」的来源，所以只作最后手段。
fn read_clipboard() -> Option<String> {
  if has_x11() {
    if has("xclip")
      && let Some((true, out)) =
        run_capture("xclip", &["-selection", "clipboard", "-o"], CMD_TIMEOUT)
      && let Some(text) = clipboard_bytes_to_text(out)
    {
      return Some(text);
    }
    if has("xsel")
      && let Some((true, out)) = run_capture("xsel", &["-b", "-o"], CMD_TIMEOUT)
      && let Some(text) = clipboard_bytes_to_text(out)
    {
      return Some(text);
    }
  }
  if has_wayland()
    && has("wl-paste")
    && let Some((true, out)) =
      run_capture("wl-paste", &["--no-newline", "--type", "text"], CMD_TIMEOUT)
    && let Some(text) = clipboard_bytes_to_text(out)
  {
    return Some(text);
  }
  None
}

/// 写剪贴板（CLIPBOARD + PRIMARY）。
///
/// **优先 X11（xclip/xsel）**：mutter 会把 X11 选区双向桥接到 Wayland，
/// 不会像 wl-copy 那样弹「wl-clipboard」窗口、也不会因焦点保护而不生效。
pub fn copy_text(text: &str) -> Result<(), String> {
  // 新一次写入：让还在等待的「还原旧剪贴板」失效
  CLIP_GEN.fetch_add(1, Ordering::SeqCst);

  let mut errors: Vec<String> = Vec::new();

  if has_x11() {
    if has("xclip") {
      match run_quiet("xclip", &["-selection", "clipboard", "-i"], Some(text), CMD_TIMEOUT) {
        Ok(()) => {
          // 同时写 PRIMARY：Shift+Insert 在部分应用里读 PRIMARY
          let _ = run_quiet("xclip", &["-selection", "primary", "-i"], Some(text), CMD_TIMEOUT);
          return Ok(());
        }
        Err(e) => errors.push(e),
      }
    }
    if has("xsel") {
      match run_quiet("xsel", &["-i", "-b"], Some(text), CMD_TIMEOUT) {
        Ok(()) => {
          let _ = run_quiet("xsel", &["-i", "-p"], Some(text), CMD_TIMEOUT);
          return Ok(());
        }
        Err(e) => errors.push(e),
      }
    }
  }

  // 纯 Wayland（没有 XWayland）时的最后手段：会短暂弹窗，但至少能用
  if has_wayland() && has("wl-copy") {
    match run_quiet("wl-copy", &["--type", "text/plain"], Some(text), CMD_TIMEOUT) {
      Ok(()) => {
        let _ = run_quiet("wl-copy", &["--primary", "--type", "text/plain"], Some(text), CMD_TIMEOUT);
        return Ok(());
      }
      Err(e) => errors.push(e),
    }
  }

  Err(if errors.is_empty() {
    "没有可用的剪贴板工具（安装 xclip 或 xsel）".into()
  } else {
    format!("剪贴板写入失败：{}", errors.join("；"))
  })
}
