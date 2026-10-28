// 说 · Shuo — 模型一键下载（GitHub Release → ~/.shuo/models/）
//
// 缺模型时 UI 一键拉取微调模型包（LM + mmproj + shuo-model.json 标记，
// 提示词可控输出风格）到 ~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/，
// 完成后自动重载引擎。支持断点续传（.part + Range）、大小校验、400ms 节流进度事件。
//
// TLS 证书走**系统信任库**（ureq platform-verifier → rustls-platform-verifier）：
// 企业/本地代理做 TLS 拦截时，其 CA 只要装进操作系统即可正常下载；不依赖
// 编译期烘焙的 webpki-roots。代理仍遵循 HTTPS_PROXY/ALL_PROXY 环境变量。
//
// ⚠ 下载源说明（给上游维护者）：
//   当前 RELEASE_REPO 指向贡献者 fork 的 pre-release（过渡期托管）。
//   合并后请改为上游自己的仓库/Release（资产文件名与目录约定保持不变即可）：
//     1. 在自己的仓库创建 Release（tag 任意），上传同名三个资产；
//     2. 修改下方 RELEASE_REPO / RELEASE_TAG 两个常量；
//     3. （可选）SHA256SUMS 一并上传，未来可加校验。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// 下载源仓库（owner/repo）——过渡期指向贡献者 fork，上游合并后改这里
pub const RELEASE_REPO: &str = "OrientCOMPASS/shuo";
/// 下载源 Release tag（pre-release 亦可）
pub const RELEASE_TAG: &str = "model-tuned-1.7B-pre";
/// 需要落盘的资产（文件名与 Release 一致；shuo-model.json 是微调包标记，
/// src-tauri/src/models.rs 据此识别「提示词可当任务指令」的微调包）
pub const ASSETS: [&str; 3] = ["tuned-Q4_K_M.gguf", "mmproj-tuned-q8.gguf", "shuo-model.json"];
/// 落盘目录名（models.rs 的 GGUF 微调包探测约定）
pub const DIR_NAME: &str = crate::models::GGUF_DIR_NAME_TUNED;

/// 共享 HTTP Agent：TLS 用**系统证书库**校验（platform-verifier）。
/// ureq 的 crate 级便捷函数（`ureq::get`）不走自定义 TLS 配置，
/// 所以本模块所有请求都经由此 Agent（顺带获得连接复用）。
fn agent() -> &'static ureq::Agent {
  static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
  AGENT.get_or_init(|| {
    use ureq::tls::{RootCerts, TlsConfig};
    ureq::Agent::config_builder()
      .tls_config(
        TlsConfig::builder()
          .root_certs(RootCerts::PlatformVerifier)
          .build(),
      )
      .build()
      .new_agent()
  })
}

static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// 下载进度事件（前端 `model-download` 监听）
#[derive(Clone, Serialize)]
pub struct Progress {
  /// preparing / downloading / done / error
  pub phase: String,
  pub asset: String,
  pub asset_received: u64,
  pub asset_total: u64,
  pub assets_done: usize,
  pub assets_total: usize,
  pub error: Option<String>,
}

fn emit(app: &AppHandle, p: Progress) {
  let _ = app.emit("model-download", p);
}

pub fn target_dir() -> PathBuf {
  crate::config::user_dir().join("models").join(DIR_NAME)
}

pub fn release_url() -> String {
  format!("https://github.com/{RELEASE_REPO}/releases/tag/{RELEASE_TAG}")
}

pub fn is_downloading() -> bool {
  DOWNLOADING.load(Ordering::SeqCst)
}

/// 本地文件状态（UI 展示 / 恢复按钮态用）
pub fn local_state() -> Vec<(String, bool, u64)> {
  ASSETS
    .iter()
    .map(|n| {
      let p = target_dir().join(n);
      (n.to_string(), p.is_file(), p.metadata().map(|m| m.len()).unwrap_or(0))
    })
    .collect()
}

fn get_json(url: &str) -> Result<serde_json::Value, String> {
  let resp = agent()
    .get(url)
    .header("User-Agent", "shuo-desktop")
    .call()
    .map_err(|e| e.to_string())?;
  serde_json::from_reader(resp.into_body().into_reader())
    .map_err(|e| format!("解析 JSON 失败：{e}"))
}

/// 查询 Release 元数据，按 ASSETS 顺序返回 (名字, 大小, 下载直链)。
fn fetch_release_assets() -> Result<Vec<(String, u64, String)>, String> {
  let url = format!(
    "https://api.github.com/repos/{RELEASE_REPO}/releases/tags/{RELEASE_TAG}"
  );
  let v = get_json(&url).map_err(|e| format!("查询 Release 失败：{e}"))?;
  let assets = v
    .get("assets")
    .and_then(|a| a.as_array())
    .ok_or_else(|| format!("Release {RELEASE_TAG} 没有资产（检查 {RELEASE_REPO}）"))?;
  let mut out = Vec::new();
  for name in ASSETS {
    let a = assets
      .iter()
      .find(|a| a.get("name").and_then(|n| n.as_str()) == Some(name))
      .ok_or_else(|| format!("Release 缺少资产 {name}"))?;
    let url = a
      .get("browser_download_url")
      .and_then(|u| u.as_str())
      .filter(|u| !u.is_empty())
      .ok_or_else(|| format!("资产 {name} 缺少下载地址"))?;
    out.push((name.to_string(), a.get("size").and_then(|s| s.as_u64()).unwrap_or(0), url.to_string()));
  }
  Ok(out)
}

fn download_asset(
  app: &AppHandle,
  name: &str,
  size: u64,
  url: &str,
  dir: &Path,
  idx: usize,
  total: usize,
) -> Result<(), String> {
  let dest = dir.join(name);
  if dest.is_file() && size > 0 && dest.metadata().is_ok_and(|m| m.len() == size) {
    // 已完整存在（大小一致）：跳过
    emit(app, Progress {
      phase: "downloading".into(), asset: name.into(),
      asset_received: size, asset_total: size,
      assets_done: idx + 1, assets_total: total, error: None,
    });
    return Ok(());
  }
  let part = dir.join(format!("{name}.part"));
  let mut have = part.metadata().map(|m| m.len()).unwrap_or(0);
  if size > 0 && have > size {
    let _ = std::fs::remove_file(&part); // .part 比目标还大：脏文件，重来
    have = 0;
  }
  let mut req = agent().get(url).header("User-Agent", "shuo-desktop");
  if have > 0 {
    req = req.header("Range", format!("bytes={have}-")); // 断点续传
  }
  let resp = req.call().map_err(|e| format!("下载 {name} 失败：{e}"))?;
  let status = resp.status();
  let resume = have > 0 && status == 206;
  if status != 200 && status != 206 {
    return Err(format!("下载 {name} 失败：HTTP {status}"));
  }
  if !resume {
    have = 0;
  }
  let mut f = std::fs::OpenOptions::new()
    .create(true)
    .write(true)
    .append(resume)
    .truncate(!resume)
    .open(&part)
    .map_err(|e| format!("无法写入 {}：{e}", part.display()))?;
  let mut reader = resp.into_body().into_reader();
  let mut buf = vec![0u8; 256 * 1024];
  let mut last = Instant::now();
  loop {
    let n = reader.read(&mut buf).map_err(|e| format!("读取 {name} 中断：{e}（可重试续传）"))?;
    if n == 0 {
      break;
    }
    f.write_all(&buf[..n]).map_err(|e| format!("写入 {name} 失败：{e}（磁盘空间？）"))?;
    have += n as u64;
    if last.elapsed().as_millis() >= 400 {
      last = Instant::now();
      emit(app, Progress {
        phase: "downloading".into(), asset: name.into(),
        asset_received: have, asset_total: size,
        assets_done: idx, assets_total: total, error: None,
      });
    }
  }
  f.flush().map_err(|e| format!("写入 {name} 失败：{e}"))?;
  drop(f);
  if size > 0 && have != size {
    return Err(format!("{name} 大小不符：{have}/{size}（可重试，自动续传）"));
  }
  std::fs::rename(&part, &dest).map_err(|e| format!("{name} 重命名失败：{e}"))?;
  emit(app, Progress {
    phase: "downloading".into(), asset: name.into(),
    asset_received: have, asset_total: size.max(have),
    assets_done: idx + 1, assets_total: total, error: None,
  });
  Ok(())
}

fn run_download(app: &AppHandle) -> Result<PathBuf, String> {
  emit(app, Progress {
    phase: "preparing".into(), asset: String::new(),
    asset_received: 0, asset_total: 0,
    assets_done: 0, assets_total: ASSETS.len(), error: None,
  });
  let assets = fetch_release_assets()?;
  let dir = target_dir();
  std::fs::create_dir_all(&dir).map_err(|e| format!("无法创建 {}：{e}", dir.display()))?;
  let total = assets.len();
  for (i, (name, size, url)) in assets.iter().enumerate() {
    crate::log_line(&format!("下载模型资产：{name}（{size} 字节，源 {}）", release_url()));
    download_asset(app, name, *size, url, &dir, i, total)?;
  }
  Ok(dir)
}

/// 启动下载（后台线程，立即返回；同一时刻只允许一个下载）。
/// 完成后自动重载模型引擎（`AppState::reload_engine_full`）。
pub fn start(app: AppHandle, state: Arc<crate::state::AppState>) -> Result<(), String> {
  if DOWNLOADING.swap(true, Ordering::SeqCst) {
    return Err("已有下载在进行中".into());
  }
  std::thread::Builder::new()
    .name("shuo-model-dl".into())
    .spawn(move || {
      let result = run_download(&app);
      DOWNLOADING.store(false, Ordering::SeqCst);
      match result {
        Ok(dir) => {
          crate::log_line(&format!("模型下载完成：{}", dir.display()));
          emit(&app, Progress {
            phase: "done".into(), asset: String::new(),
            asset_received: 0, asset_total: 0,
            assets_done: ASSETS.len(), assets_total: ASSETS.len(), error: None,
          });
          state.reload_engine_full();
        }
        Err(e) => {
          crate::log_line(&format!("模型下载失败：{e}"));
          emit(&app, Progress {
            phase: "error".into(), asset: String::new(),
            asset_received: 0, asset_total: 0,
            assets_done: 0, assets_total: ASSETS.len(), error: Some(e),
          });
        }
      }
    })
    .map_err(|e| {
      DOWNLOADING.store(false, Ordering::SeqCst);
      format!("无法启动下载线程：{e}")
    })?;
  Ok(())
}
