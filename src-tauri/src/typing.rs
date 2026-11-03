// 说 · Shuo — 自动输入（把识别结果打到当前焦点窗口）
//
// Windows：SendInput + KEYEVENTF_UNICODE（不碰剪贴板，终端 / Vim 也能用，支持非 BMP 字符）。
// Linux：剪贴板（xclip/xsel，XWayland）＋ 粘贴快捷键；wl-copy / 门户 / 外部工具都带超时。
//
// 上屏在独立线程（state.rs 的 shuo-output）里串行执行，绝不阻塞录音与识别。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use tauri::AppHandle;

#[cfg(windows)]
#[path = "typing_windows.rs"]
mod imp;
#[cfg(target_os = "linux")]
#[path = "typing_linux.rs"]
mod imp;
#[cfg(not(any(windows, target_os = "linux")))]
#[path = "typing_other.rs"]
mod imp;

/// 上屏方式：由触发录音的热键决定（主热键 Ctrl+V / 终端热键 Ctrl+Shift+V）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PasteMode {
  #[default]
  CtrlV,
  CtrlShiftV,
}

/// 输入文本；失败返回给用户看的原因（不静默）。
pub fn type_text(_app: &AppHandle, text: &str, mode: PasteMode) -> Result<(), String> {
  imp::type_text(text, mode)
}

/// 复制到剪贴板（结果卡片「复制」按钮的 Rust 兜底）。
pub fn copy_text(text: &str) -> Result<(), String> {
  imp::copy_text(text)
}
