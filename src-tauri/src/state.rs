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

use crate::asr::AsrPipeline;
use crate::audio::RecorderHandle;
use crate::config::{Config, HistoryItem};
use crate::denoise::Denoiser;
use crate::gguf_asr::GgufAsr;
use crate::models::{self, Backend, ModelPaths};

/// ASR 推理引擎：ONNX（ort，Daumee 0.6B 包）或 GGUF（llama.cpp，1.7B）。
/// 两条路径对外同接口（transcribe/transcribe_single → AsrResult），worker 无差别调度。
pub enum AsrEngine {
  Onnx(AsrPipeline),
  Gguf(GgufAsr),
}

impl AsrEngine {
  pub fn transcribe(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    max_new_tokens: usize,
    chunk_sec: usize,
  ) -> Result<crate::asr::AsrResult, String> {
    match self {
      AsrEngine::Onnx(p) => p.transcribe(wav, language, context, max_new_tokens, chunk_sec),
      AsrEngine::Gguf(p) => p.transcribe(wav, language, context, max_new_tokens, chunk_sec),
    }
  }

  pub fn transcribe_single(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    max_new_tokens: usize,
  ) -> Result<crate::asr::AsrResult, String> {
    match self {
      AsrEngine::Onnx(p) => p.transcribe_single(wav, language, context, max_new_tokens),
      AsrEngine::Gguf(p) => p.transcribe_single(wav, language, context, max_new_tokens),
    }
  }
}

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
  /// 当前 ASR 推理后端："onnx" / "gguf" / ""（未加载）
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
  pub asr: Mutex<Option<AsrEngine>>,
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
      asr_dir: paths.asr.as_ref().map(|p| p.display().to_string()),
      error: None,
      denoise_ready: false,
      style_model: paths.asr_tuned,
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
    let cur_backend = self.status.lock().unwrap().asr_backend.clone();
    self.set_status(|s| {
      s.asr_dir = paths.asr.as_ref().map(|p| p.display().to_string());
      s.style_model = if cur_backend == "gguf" { paths.gguf_tuned } else { paths.asr_tuned };
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

      let backend = models::pick_backend(&cfg, &paths);
      let Some(backend) = backend else {
        state.set_status(|s| {
          s.loading = false;
          s.error = Some(format!(
            "找不到 ASR 模型（ONNX：Qwen3-ASR-0.6B-ONNX-CPU[-tuned]，GGUF：Qwen3-ASR-1.7B-GGUF[-tuned]）。已尝试：\n{}",
            paths.tried.join("\n")
          ));
        });
        return;
      };

      state.set_status(|s| {
        s.loading = true;
        s.error = None;
      });

      // 按所选后端加载；显式选择的后端失败时回退另一个可用后端（日志说明）
      let mut loaded: Option<(Backend, AsrEngine, String)> = None;
      let mut order = vec![backend];
      let other = if backend == Backend::Gguf { Backend::Onnx } else { Backend::Gguf };
      order.push(other);
      let mut last_err = String::new();
      for b in order {
        match b {
          Backend::Gguf => {
            let Some(g) = paths.gguf.clone() else { continue };
            crate::log_line(&format!("加载 GGUF ASR：{} + {}", g.lm.display(), g.mmproj.display()));
            match GgufAsr::load(&g.lm, &g.mmproj, threads, 512) {
              Ok(p) => {
                let dir = g.lm.parent().map(|d| d.display().to_string()).unwrap_or_default();
                loaded = Some((Backend::Gguf, AsrEngine::Gguf(p), dir));
                break;
              }
              Err(e) => {
                last_err = e;
                crate::log_line(&format!("GGUF ASR 加载失败：{last_err}"));
              }
            }
          }
          Backend::Onnx => {
            let Some(dir) = paths.asr.clone() else { continue };
            crate::log_line(&format!("加载 ONNX ASR：{}", dir.display()));
            match AsrPipeline::load(&dir, threads) {
              Ok(p) => {
                loaded = Some((Backend::Onnx, AsrEngine::Onnx(p), dir.display().to_string()));
                break;
              }
              Err(e) => {
                last_err = e;
                crate::log_line(&format!("ONNX ASR 加载失败：{last_err}"));
              }
            }
          }
        }
      }

      let Some((active, engine, dir)) = loaded else {
        state.set_status(|s| {
          s.loading = false;
          s.error = Some(format!("ASR 模型加载失败：{last_err}"));
        });
        return;
      };
      *state.asr.lock().unwrap() = Some(engine);
      let tuned = models::tuned_for(active, &paths);
      crate::log_line(&format!(
        "ASR 模型就绪（后端 {}，{}）",
        match active { Backend::Gguf => "gguf/llama.cpp", Backend::Onnx => "onnx/ort" },
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
        s.asr_backend = match active { Backend::Gguf => "gguf".into(), Backend::Onnx => "onnx".into() };
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

fn transcribe(
  state: &Arc<AppState>,
  cfg: &Config,
  job: Job,
) -> Result<crate::asr::AsrResult, String> {
  let mut samples = job.samples;
  if job.rate != 16000 {
    samples = crate::dsp::resample(&samples, job.rate, 16000)?;
  }

  // 降噪路径：16k → 48k → 降噪 → 16k
  if cfg.denoise {
    let mut guard = state.denoiser.lock().unwrap();
    if let Some(d) = guard.as_mut() {
      let up = crate::dsp::resample(&samples, 16000, 48000)?;
      let den = d.denoise(&up)?;
      samples = crate::dsp::resample(&den, 48000, 16000)?;
    }
  }

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
  if cfg.denoise {
    pipeline.transcribe_single(&samples, language, &context, 512)
  } else {
    pipeline.transcribe(&samples, language, &context, 512, 30)
  }
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
