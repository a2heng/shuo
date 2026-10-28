// 说 · Shuo — 模型一键下载（ModelScope 模型仓库 → ~/.shuo/models/）
//
// 缺模型时 UI 一键拉取微调模型包（LM + mmproj + shuo-model.json 标记，
// 提示词可控输出风格）到 ~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/，
// 完成后自动重载引擎。支持断点续传（.part + Range）、大小 + SHA256 校验、
// 400ms 节流进度事件。
//
// TLS 证书走**系统信任库**（ureq platform-verifier → rustls-platform-verifier）：
// 企业/本地代理做 TLS 拦截时，其 CA 只要装进操作系统即可正常下载；不依赖
// 编译期烘焙的 webpki-roots。代理仍遵循 HTTPS_PROXY/ALL_PROXY 环境变量。
//
// 下载源说明（为什么是 ModelScope 而不是 GitHub Release）：
//   * 国内直连 modelscope.cn 单连接约 10 MB/s；GitHub Release 直连常被限到
//     几十 KB/s，HuggingFace 需要代理或 hf-mirror（镜像站只读，不能上传）；
//   * 仓库公开，下载不需要登录/Token。
//   换源只需改下方 REPO / REVISION（资产文件名与落盘目录约定保持不变）；
//   企业/自建镜像可用 SHUO_MODEL_ENDPOINT 覆盖域名，路径约定
//   `<endpoint>/models/<REPO>/resolve/<REVISION>/<file>`。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

/// 下载源仓库（ModelScope owner/name，公开仓库，无需 Token）
pub const REPO: &str = "a2heng/shuo-asr-1.7b-gguf";
/// 下载源分支/版本（ModelScope 默认 master）
pub const REVISION: &str = "master";
/// 需要落盘的资产（文件名与仓库一致；shuo-model.json 是微调包标记，
/// src-tauri/src/models.rs 据此识别「提示词可当任务指令」的微调包）
pub const ASSETS: [&str; 3] = ["tuned-Q4_K_M.gguf", "mmproj-tuned-q8.gguf", "shuo-model.json"];
/// 落盘目录名（models.rs 的 GGUF 微调包探测约定）
pub const DIR_NAME: &str = crate::models::GGUF_DIR_NAME_TUNED;

/// 下载源域名：默认 ModelScope，可用 SHUO_MODEL_ENDPOINT 覆盖（镜像 / 企业代理）
pub fn endpoint() -> String {
  normalize_endpoint(std::env::var("SHUO_MODEL_ENDPOINT").ok().as_deref())
}

/// 归一化镜像地址（去空白 + 去掉结尾 `/`；空值回退默认）
fn normalize_endpoint(raw: Option<&str>) -> String {
  raw
    .map(|v| v.trim().trim_end_matches('/').to_string())
    .filter(|v| !v.is_empty())
    .unwrap_or_else(|| "https://modelscope.cn".to_string())
}

/// 单个资产的下载直链（ModelScope resolve 支持 Range，可续传）
fn resolve_url(name: &str) -> String {
  format!("{}/models/{REPO}/resolve/{REVISION}/{name}", endpoint())
}

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

/// 模型仓库页（UI「来源页」链接用）
pub fn repo_url() -> String {
  format!("{}/models/{REPO}", endpoint())
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

/// 查询仓库文件清单，按 ASSETS 顺序返回 (名字, 大小, sha256, 下载直链)。
/// ModelScope 的 `repo/files` 一次返回全部条目（含 Size / Sha256），
/// 公开仓库不需要登录，也不依赖 Release API。
fn fetch_assets() -> Result<Vec<(String, u64, String, String)>, String> {
  let url = format!(
    "{}/api/v1/models/{REPO}/repo/files?Revision={REVISION}&Recursive=true",
    endpoint()
  );
  let v = get_json(&url).map_err(|e| format!("查询模型仓库失败：{e}"))?;
  if let Some(code) = v.get("Code").and_then(|c| c.as_u64()) {
    if code != 200 {
      return Err(format!("查询模型仓库失败：Code {code}（检查 {REPO}）"));
    }
  }
  let files = v
    .get("Data")
    .and_then(|d| d.get("Files"))
    .and_then(|f| f.as_array())
    .ok_or_else(|| format!("模型仓库 {REPO} 没有文件清单"))?;
  let mut out = Vec::new();
  for name in ASSETS {
    let f = files
      .iter()
      .find(|f| f.get("Name").and_then(|n| n.as_str()) == Some(name))
      .ok_or_else(|| format!("模型仓库缺少资产 {name}"))?;
    let size = f.get("Size").and_then(|s| s.as_u64()).unwrap_or(0);
    let sha = f.get("Sha256").and_then(|s| s.as_str()).unwrap_or("").to_string();
    out.push((name.to_string(), size, sha, resolve_url(name)));
  }
  Ok(out)
}

/// 文件 SHA256（小写十六进制），用于下载后校验
fn sha256_file(path: &Path) -> Result<String, String> {
  use sha2::{Digest, Sha256};
  let mut f = std::fs::File::open(path).map_err(|e| format!("无法读取 {}：{e}", path.display()))?;
  let mut hasher = Sha256::new();
  let mut buf = vec![0u8; 1024 * 1024];
  loop {
    let n = f
      .read(&mut buf)
      .map_err(|e| format!("读取 {} 失败：{e}", path.display()))?;
    if n == 0 {
      break;
    }
    hasher.update(&buf[..n]);
  }
  Ok(format!("{:x}", hasher.finalize()))
}

fn download_asset(
  app: &AppHandle,
  name: &str,
  size: u64,
  url: &str,
  sha: &str,
  dir: &Path,
  idx: usize,
  total: usize,
) -> Result<(), String> {
  let dest = dir.join(name);
  if dest.is_file() && size > 0 && dest.metadata().is_ok_and(|m| m.len() == size) {
    // 已完整存在（大小一致）：SHA256 也对得上才跳过，否则重下
    let ok = sha.is_empty() || sha256_file(&dest)?.eq_ignore_ascii_case(sha);
    if ok {
      emit(app, Progress {
        phase: "downloading".into(), asset: name.into(),
        asset_received: size, asset_total: size,
        assets_done: idx + 1, assets_total: total, error: None,
      });
      return Ok(());
    }
    let _ = std::fs::remove_file(&dest);
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
  if !sha.is_empty() {
    let got = sha256_file(&part)?;
    if !got.eq_ignore_ascii_case(sha) {
      let _ = std::fs::remove_file(&part);
      return Err(format!("{name} 校验失败：SHA256 不符，已删除（可重试）"));
    }
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
  let assets = fetch_assets()?;
  let dir = target_dir();
  std::fs::create_dir_all(&dir).map_err(|e| format!("无法创建 {}：{e}", dir.display()))?;

  // 已有完整文件（大小 + SHA256 都对）就不再下载：直接报「已完成」，
  // 只重载引擎（换模型包/修过文件时也能刷新）。
  let mut missing: Vec<usize> = Vec::new();
  for (i, (name, size, sha, _)) in assets.iter().enumerate() {
    let dest = dir.join(name);
    let ok = dest.is_file()
      && dest.metadata().is_ok_and(|m| m.len() == *size)
      && (sha.is_empty() || sha256_file(&dest).is_ok_and(|h| h.eq_ignore_ascii_case(sha)));
    if !ok {
      missing.push(i);
    }
  }
  if missing.is_empty() {
    crate::log_line(&format!(
      "模型已存在且 SHA256 校验通过（{} 个文件），跳过下载：{}",
      assets.len(),
      dir.display()
    ));
    return Ok(dir);
  }
  if missing.len() < assets.len() {
    crate::log_line(&format!(
      "本地已有 {} 个资产校验通过，只下缺失/损坏的 {} 个",
      assets.len() - missing.len(),
      missing.len()
    ));
  }

  let total = assets.len();
  for i in missing {
    let (name, size, sha, url) = &assets[i];
    crate::log_line(&format!("下载模型资产：{name}（{size} 字节，源 {}）", repo_url()));
    download_asset(app, name, *size, url, sha, &dir, i, total)?;
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
          crate::log_line(&format!("模型已就绪：{}", dir.display()));
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn endpoint_normalisation() {
    assert_eq!(normalize_endpoint(None), "https://modelscope.cn");
    assert_eq!(normalize_endpoint(Some("   ")), "https://modelscope.cn");
    assert_eq!(
      normalize_endpoint(Some(" https://mirror.example.com/ch/ ")),
      "https://mirror.example.com/ch"
    );
  }

  #[test]
  fn asset_urls_follow_repo_convention() {
    let url = resolve_url("shuo-model.json");
    assert!(url.ends_with("/models/a2heng/shuo-asr-1.7b-gguf/resolve/master/shuo-model.json"), "{url}");
    assert!(repo_url().ends_with("/models/a2heng/shuo-asr-1.7b-gguf"), "{}", repo_url());
  }

  /// 真实网络用例：`cargo test --release -- --ignored modelscope_assets`
  /// 校验 ModelScope 清单里的三个资产都在、大小/SHA256 齐全，并实际下载最小的
  /// `shuo-model.json` 比对哈希（不拉 GB 级权重）。
  #[test]
  #[ignore = "network: 访问 modelscope.cn"]
  fn modelscope_assets_listed() {
    let assets = fetch_assets().expect("查询 ModelScope 模型仓库失败");
    assert_eq!(assets.len(), ASSETS.len());
    for (name, size, sha, url) in &assets {
      assert!(*size > 0, "{name} 大小未知");
      assert_eq!(sha.len(), 64, "{name} 缺少 SHA256");
      assert!(url.ends_with(name.as_str()), "{name} 下载地址异常：{url}");
    }

    let (_, _, sha, url) = assets
      .iter()
      .find(|a| a.0 == "shuo-model.json")
      .expect("清单里没有 shuo-model.json");
    let mut body = agent()
      .get(url)
      .call()
      .expect("下载 shuo-model.json 失败")
      .into_body();
    let bytes = body.read_to_vec().expect("读取 shuo-model.json 失败");
    use sha2::{Digest, Sha256};
    let got = format!("{:x}", Sha256::digest(&bytes));
    assert!(got.eq_ignore_ascii_case(sha), "SHA256 不匹配：{got} != {sha}");
    assert!(String::from_utf8_lossy(&bytes).contains("\"kind\": \"shuo-model\""));
  }
}
