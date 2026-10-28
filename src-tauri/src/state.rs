// 说 · Shuo — 应用状态与识别线程
//
// 单一线程串行处理识别任务（ASR 输出即最终文本），与 Python 版的队列语义一致。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::audio::RecorderHandle;
use crate::config::{Config, HistoryItem};
use crate::denoise::Denoiser;
use crate::gguf_asr::GgufAsr;
use crate::models::{self, ModelPaths};

pub struct Job {
  pub samples: Vec<f32>,
  pub rate: u32,
  pub duration_s: f64,
  pub paste_mode: crate::typing::PasteMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct Status {
  pub loading: bool,
  pub asr_ready: bool,
  pub asr_dir: Option<String>,
  pub error: Option<String>,
  pub denoise_ready: bool,
  /// 当前模型包是否为微调包（system 段提示词可当任务指令，控制输出风格）
  pub style_model: bool,
  /// ASR 推理后端（唯一实现：llama.cpp/GGUF；保留字段给 UI chip）
  pub asr_backend: String,
}

#[derive(Clone)]
pub struct Sink {
  app: AppHandle,
  tx: Sender<Job>,
}

impl Sink {
  pub fn emit_record(&self, active: bool, duration_s: f64) {
    let _ = self.app.emit("rec", RecEvent { active, duration_s });
    if active {
      crate::hotkey::set_recording(true);
      crate::set_busy(&self.app, Some(crate::BusyMode::Recording));
    } else if duration_s >= 0.1 {
      // 松开后马上进入「识别中」，避免闪一下消失
      crate::set_busy(&self.app, Some(crate::BusyMode::Processing));
    } else {
      crate::set_busy(&self.app, None);
    }
  }

  pub fn submit(&self, job: Job) {
    if self.tx.send(job).is_err() {
      crate::log_line("识别队列已关闭");
    }
  }
}

#[derive(Clone, Serialize)]
struct RecEvent {
  active: bool,
  duration_s: f64,
}

/// 热键状态上报（前端用来显示「全局热键：F2（门户）」这类信息）。
pub fn emit_hotkey(
  state: &AppState,
  ok: bool,
  backend: &str,
  spec: &str,
  error: Option<&str>,
) {
  let _ = state.app.emit(
    "hotkey",
    serde_json::json!({
      "ok": ok,
      "backend": backend,
      "spec": spec,
      "display": crate::hotkey::normalize_spec(spec),
      "error": error,
    }),
  );
}

pub struct AppState {
  pub app: AppHandle,
  pub cfg: Mutex<Config>,
  pub paths: Mutex<ModelPaths>,
  pub asr: Mutex<Option<GgufAsr>>,
  pub denoiser: Mutex<Option<Denoiser>>,
  pub status: Mutex<Status>,
  pub recorder: Mutex<Option<RecorderHandle>>,
  pub hotkey: Mutex<Option<crate::hotkey::Hotkey>>,
  pub history: Mutex<Vec<HistoryItem>>,
  pub sink: Sink,
  next_id: AtomicU64,
}

impl AppState {
  pub fn new(app: AppHandle, cfg: Config) -> Arc<Self> {
    let paths = models::resolve(&cfg);
    let (tx, rx) = channel::<Job>();
    let status = Status {
      loading: true,
      asr_ready: false,
      asr_dir: paths
        .gguf
        .as_ref()
        .and_then(|g| g.lm.parent())
        .map(|d| d.display().to_string()),
      error: None,
      denoise_ready: false,
      style_model: paths.gguf_tuned,
      asr_backend: String::new(),
    };
    let state = Arc::new(Self {
      app: app.clone(),
      cfg: Mutex::new(cfg),
      paths: Mutex::new(paths),
      asr: Mutex::new(None),
      denoiser: Mutex::new(None),
      status: Mutex::new(status),
      recorder: Mutex::new(None),
      hotkey: Mutex::new(None),
      history: Mutex::new(crate::config::load_history()),
      sink: Sink { app, tx },
      next_id: AtomicU64::new(1),
    });
    spawn_loader(state.clone());
    spawn_worker(state.clone(), rx);
    state
  }

  pub fn config(&self) -> Config {
    self.cfg.lock().unwrap().clone()
  }

  pub fn emit_status(&self) {
    let s = self.status.lock().unwrap().clone();
    let _ = self.app.emit("engine-status", s);
  }

  pub fn set_status<F: FnOnce(&mut Status)>(&self, f: F) {
    {
      let mut s = self.status.lock().unwrap();
      f(&mut s);
    }
    self.emit_status();
  }

  pub fn next_job_id(&self) -> u64 {
    self.next_id.fetch_add(1, Ordering::Relaxed)
  }

  /// 配置变更后重新解析模型路径（引擎重载见 `reload_engine_full`）。
  pub fn reload_models(self: &Arc<Self>) {
    let cfg = self.config();
    let paths = models::resolve(&cfg);
    {
      let mut p = self.paths.lock().unwrap();
      *p = paths.clone();
    }
    self.set_status(|s| {
      s.asr_dir = paths
        .gguf
        .as_ref()
        .and_then(|g| g.lm.parent())
        .map(|d| d.display().to_string());
      s.style_model = paths.gguf_tuned;
    });
  }

  /// 完整重载：重新解析路径并重启加载线程（「重新加载」按钮与模型下载完成后调用）。
  /// 加载器完成时会原子替换 `asr` 槽位（Mutex 保护，worker 正在识别则等锁释放）。
  pub fn reload_engine_full(self: &Arc<Self>) {
    self.reload_models();
    if self.status.lock().unwrap().loading {
      return; // 已在加载，避免双份加载
    }
    spawn_loader(self.clone());
  }

  /// 按配置（重新）启动录音。
  pub fn restart_recorder(self: &Arc<Self>) -> Result<(), String> {
    let old = self.recorder.lock().unwrap().take();
    if let Some(h) = old {
      let _ = h.stop_signal();
    }
    let cfg = self.config();
    match crate::audio::start_recorder(self.sink.clone(), cfg.mic_device.clone()) {
      Ok(h) => {
        crate::log_line(&format!("使用麦克风：{} ({} Hz)", h.device, h.rate));
        *self.recorder.lock().unwrap() = Some(h);
        let _ = self.app.emit(
          "recorder",
          serde_json::json!({ "ok": true, "error": serde_json::Value::Null }),
        );
        Ok(())
      }
      Err(e) => {
        crate::log_line(&format!("麦克风不可用：{e}"));
        let _ = self.app.emit(
          "recorder",
          serde_json::json!({ "ok": false, "error": e }),
        );
        Err(e)
      }
    }
  }

  pub fn press(&self, mode: crate::typing::PasteMode) {
    if let Some(h) = self.recorder.lock().unwrap().as_ref() {
      h.press(mode);
    }
  }

  pub fn release(&self) {
    if let Some(h) = self.recorder.lock().unwrap().as_ref() {
      h.release();
    }
  }

  pub fn start_hotkey(self: &Arc<Self>) {
    let cfg = self.config();
    let old = self.hotkey.lock().unwrap().take();
    if let Some(h) = old {
      h.stop();
    }
    // 主热键 = Ctrl+V 上屏；终端热键 = Ctrl+Shift+V 上屏
    let mut list = Vec::new();
    if cfg.hotkey_enabled {
      list.push((cfg.hotkey.clone(), crate::typing::PasteMode::CtrlV));
    }
    if cfg.hotkey_terminal_enabled {
      list.push((cfg.hotkey_terminal.clone(), crate::typing::PasteMode::CtrlShiftV));
    }
    let h = crate::hotkey::start(self.clone(), list);
    *self.hotkey.lock().unwrap() = Some(h);
  }
}

fn spawn_loader(state: Arc<AppState>) {
  std::thread::Builder::new()
    .name("shuo-loader".into())
    .spawn(move || {
      let cfg = state.config();
      let paths = state.paths.lock().unwrap().clone();
      let threads = crate::threads();

      // 唯一推理路径：llama.cpp/GGUF（ONNX ASR 后端已移除）
      let Some(g) = paths.gguf.clone() else {
        state.set_status(|s| {
          s.loading = false;
          s.error = Some(format!(
            "找不到 GGUF ASR 模型（Qwen3-ASR-1.7B-GGUF-tuned/ 或 Qwen3-ASR-1.7B-GGUF/）。\
             可在 设置 → 模型与引擎 一键下载。已尝试：\n{}",
            paths.tried.join("\n")
          ));
        });
        return;
      };

      state.set_status(|s| {
        s.loading = true;
        s.error = None;
      });

      crate::log_line(&format!("加载 GGUF ASR：{} + {}", g.lm.display(), g.mmproj.display()));
      let engine = match GgufAsr::load(&g.lm, &g.mmproj, threads, 512) {
        Ok(p) => p,
        Err(e) => {
          crate::log_line(&format!("GGUF ASR 加载失败：{e}"));
          state.set_status(|s| {
            s.loading = false;
            s.error = Some(format!("ASR 模型加载失败：{e}"));
          });
          return;
        }
      };

      *state.asr.lock().unwrap() = Some(engine);
      let tuned = paths.gguf_tuned;
      let dir = g.lm.parent().map(|d| d.display().to_string()).unwrap_or_default();
      crate::log_line(&format!(
        "ASR 模型就绪（后端 gguf/llama.cpp，{}）",
        if tuned { "微调包：提示词可控输出风格" } else { "基座包" }
      ));
      if !tuned && !cfg.style_context().is_empty() {
        crate::log_line(
          "提示：当前为基座模型包（缺 shuo-model.json），提示词只作识别偏置，\
           不会控制输出风格；选「逐字转写」可完全消除偏置。微调模型包可在 设置 → 模型与引擎 一键下载",
        );
      }
      state.set_status(|s| {
        s.asr_ready = true;
        s.loading = false;
        s.style_model = tuned;
        s.asr_dir = Some(dir);
        s.asr_backend = "gguf".into();
      });

      // 降噪模型（可选）
      if let Some(path) = crate::denoise_path(&state.app) {
        match Denoiser::load(&path) {
          Ok(d) => {
            *state.denoiser.lock().unwrap() = Some(d);
            crate::log_line("降噪模型就绪");
            state.set_status(|s| s.denoise_ready = true);
          }
          Err(e) => crate::log_line(&format!("降噪模型加载失败：{e}")),
        }
      }
    })
    .ok();
}

fn spawn_worker(state: Arc<AppState>, rx: std::sync::mpsc::Receiver<Job>) {
  std::thread::Builder::new()
    .name("shuo-infer".into())
    .spawn(move || {
      while let Ok(job) = rx.recv() {
        let id = state.next_job_id();
        let _ = state.app.emit("processing", id);
        crate::set_busy(&state.app, Some(crate::BusyMode::Processing));
        let cfg = state.config();
        let paste_mode = job.paste_mode;
        let result = transcribe(&state, &cfg, job);
        match result {
          Ok(asr) => {
            crate::set_busy(&state.app, None);
            let _ = state.app.emit(
              "asr",
              serde_json::json!({
                "id": id,
                "text": asr.text,
                "language": asr.language,
                "asr_s": asr.asr_s,
                "audio_s": asr.audio_s,
                "chunks": asr.chunks,
              }),
            );
            if cfg.save_history {
              crate::config::add_history(&asr.text);
              let mut h = state.history.lock().unwrap();
              h.push(crate::config::HistoryItem {
                asr_text: asr.text.clone(),
                llm_text: None,
                time: crate::config::chrono_time(),
              });
              // 与历史文件一致：内存里也最多留 500 条，避免长跑无限增长
              if h.len() > 500 {
                let cut = h.len() - 500;
                h.drain(..cut);
              }
            }
            let final_text = finalize_text(&state, &cfg, &asr.text);
            let style_model = state.status.lock().unwrap().style_model;
            let _ = state.app.emit(
              "final",
              serde_json::json!({
                "id": id,
                "asr_text": asr.text,
                "style_model": style_model,
                "text": final_text,
                "time": crate::config::chrono_time(),
              }),
            );
            if cfg.auto_type && !final_text.trim().is_empty() {
              if let Err(e) = crate::typing::type_text(&state.app, &final_text, paste_mode) {
                crate::log_line(&format!("自动输入失败：{e}"));
                let _ = state.app.emit("notice", e);
              }
            }
          }
          Err(e) => {
            crate::set_busy(&state.app, None);
            crate::log_line(&format!("识别失败：{e}"));
            let _ = state.app.emit("asr-error", serde_json::json!({ "id": id, "error": e }));
          }
        }
      }
    })
    .ok();
}

/// 采样率归一化（+ 可选降噪）：**任意设备采样率** → 16 kHz。
///
/// 降噪模型固定 48 kHz，所以先把「设备实际采样率」（16k / 44.1k / 48k / 96k …）
/// 直接重采样到 48 kHz 再降噪，最后降回 16 kHz——不是假设输入就是 48k。
/// 不降噪时只做一次 `rate → 16000`。单测覆盖 16k/44.1k/48k/96k。
fn normalize_audio(
  samples: Vec<f32>,
  rate: u32,
  denoiser: Option<&mut crate::denoise::Denoiser>,
) -> Result<Vec<f32>, String> {
  let mut samples = samples;
  let mut rate = rate;
  if let Some(d) = denoiser {
    let up = if rate != crate::denoise::SR as u32 {
      crate::dsp::resample(&samples, rate, crate::denoise::SR as u32)?
    } else {
      samples
    };
    let den = d.denoise(&up)?;
    samples = crate::dsp::resample(&den, crate::denoise::SR as u32, 16000)?;
    rate = 16000;
  }
  if rate != 16000 {
    samples = crate::dsp::resample(&samples, rate, 16000)?;
  }
  Ok(samples)
}

fn transcribe(
  state: &Arc<AppState>,
  cfg: &Config,
  job: Job,
) -> Result<crate::asr::AsrResult, String> {
  let t0 = std::time::Instant::now();
  let audio_s = job.duration_s;

  let mut guard = state.denoiser.lock().unwrap();
  let denoiser = if cfg.denoise { guard.as_mut() } else { None };
  let t_norm = std::time::Instant::now();
  let samples = normalize_audio(job.samples, job.rate, denoiser)?;
  let denoise_s = t_norm.elapsed().as_secs_f64();

  let lang_owned = cfg.asr_lang.clone();
  let language = if lang_owned == "auto" || lang_owned.is_empty() {
    None
  } else {
    Some(lang_owned.as_str())
  };

  // 输出风格：用户提示词进 chat 模板 system 段（context 通道），识别即按风格输出。
  // `verbatim` 预设 = 空 context = 原生逐字转写。基座包（无 shuo-model.json）上
  // 提示词只作识别偏置，功能不炸（日志已在加载时提示一次）。
  let context = cfg.style_context();

  let mut guard = state.asr.lock().unwrap();
  let pipeline = guard.as_mut().ok_or("ASR 模型尚未加载完成")?;
  let result = if cfg.denoise {
    pipeline.transcribe_single(&samples, language, &context, 512)
  } else {
    pipeline.transcribe(&samples, language, &context, 512, 30)
  };
  let total = t0.elapsed().as_secs_f64();
  if let Ok(ref r) = result {
    let style = if context.trim().is_empty() {
      "逐字转写（无提示词）".to_string()
    } else {
      let head: String = context.trim().chars().take(24).collect();
      format!("{head}{}（{} 字）", if context.trim().chars().count() > 24 { "…" } else { "" }, context.trim().chars().count())
    };
    crate::log_line(&format!(
      "识别完成：音频 {audio_s:.2}s，降噪 {denoise_s:.2}s，ASR {:.2}s，总 {total:.2}s（RTF {:.2}，风格：{style}）",
      r.asr_s,
      total / audio_s.max(0.01)
    ));
  }
  result
}

fn finalize_text(_state: &Arc<AppState>, cfg: &Config, text: &str) -> String {
  if cfg.language == "zh" && cfg.remove_punc {
    const PUNC: &[char] = &[
      '，', '。', '！', '？', '、', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《',
      '》', '…', '—', ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '<',
      '>', '~', '·', '～',
    ];
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
      let c = if PUNC.contains(&ch) || ch.is_whitespace() { ' ' } else { ch };
      if c == ' ' {
        if !last_space {
          out.push(' ');
        }
        last_space = true;
      } else {
        out.push(c);
        last_space = false;
      }
    }
    return out.trim().to_string();
  }
  text.trim().to_string()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn tone(rate: u32, secs: f32) -> Vec<f32> {
    let n = (rate as f32 * secs) as usize;
    (0..n)
      .map(|i| {
        let t = i as f32 / rate as f32;
        (t * 440.0 * std::f32::consts::TAU).sin() * 0.2
      })
      .collect()
  }

  /// 任意设备采样率都必须归一化到 16 kHz（16k/22.05k/32k/44.1k/48k/96k）。
  #[test]
  fn normalize_audio_any_input_rate() {
    for rate in [16000u32, 22050, 32000, 44100, 48000, 96000] {
      let out = normalize_audio(tone(rate, 1.0), rate, None).expect("resample");
      let got = out.len() as i64;
      assert!(
        (got - 16000).abs() <= 400,
        "输入 {rate} Hz → 输出 {got} 样本（期望 ≈16000）"
      );
    }
  }

  /// 带降噪：输入先按实际 rate 升/降到 48k，再降回 16k，长度仍应 ≈1 s。
  #[test]
  fn normalize_audio_with_denoiser_any_input_rate() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/denoise.onnx");
    if !path.is_file() {
      eprintln!("跳过：{} 不存在", path.display());
      return;
    }
    for rate in [16000u32, 44100, 48000] {
      let mut d = crate::denoise::Denoiser::load(&path).expect("load denoise");
      let out = normalize_audio(tone(rate, 1.0), rate, Some(&mut d)).expect("denoise");
      let got = out.len() as i64;
      assert!(
        (got - 16000).abs() <= 400,
        "降噪：输入 {rate} Hz → {got} 样本（期望 ≈16000）"
      );
    }
  }
}
