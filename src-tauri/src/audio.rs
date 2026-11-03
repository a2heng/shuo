// 说 · Shuo — 麦克风采集（cpal）
//
// 常开采集 + 0.5 s 预缓冲：按下热键/按钮时把预缓冲并入本段，松开后 500 ms 收尾
// （这段时间内再按下则继续同一段，不丢前一截）。整段在收尾时做 AGC，然后交给识别线程。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::mel;
use crate::state::{Job, Sink};
use crate::typing::PasteMode;

const PRE_BUF_SEC: f32 = 0.5;
const DEBOUNCE: Duration = Duration::from_millis(500);
const MIN_SAMPLES_SEC: f32 = 0.1;

pub struct RecorderStatus {
  pub ok: bool,
  pub device: String,
  pub rate: u32,
  pub error: Option<String>,
}

struct Shared {
  pre: VecDeque<f32>,
  pre_cap: usize,
  active: bool,
  paste_mode: PasteMode,
  buf: Vec<f32>,
  finalize_at: Option<Instant>,
}

#[derive(Debug)]
pub enum Cmd {
  Press(PasteMode),
  Release,
  Quit,
}

pub struct RecorderHandle {
  tx: Sender<Cmd>,
  pub device: String,
  pub rate: u32,
}

impl RecorderHandle {
  /// 停止采集线程（切换设备 / 退出时用）。
  pub fn stop_signal(&self) {
    let _ = self.tx.send(Cmd::Quit);
  }

  pub fn press(&self, mode: PasteMode) {
    let _ = self.tx.send(Cmd::Press(mode));
  }
  pub fn release(&self) {
    let _ = self.tx.send(Cmd::Release);
  }
}

pub fn list_input_devices() -> Vec<(String, bool)> {
  let host = cpal::default_host();
  let default_name = host.default_input_device().map(|d| dev_name(&d));
  let mut out = Vec::new();
  if let Ok(devices) = host.input_devices() {
    for d in devices {
      let name = dev_name(&d);
      if name.is_empty() || out.iter().any(|(n, _): &(String, bool)| n == &name) {
        continue;
      }
      let is_default = default_name.as_deref() == Some(name.as_str());
      out.push((name, is_default));
    }
  }
  out
}

/// 打开采集线程。`report` 用于把打开结果（设备/采样率/错误）回给调用方。
pub fn start_recorder(
  sink: Sink,
  device: Option<String>,
) -> Result<RecorderHandle, String> {
  let (tx, rx) = channel::<Cmd>();
  let (ready_tx, ready_rx) = channel::<Result<(String, u32), String>>();
  std::thread::Builder::new()
    .name("shuo-capture".into())
    .spawn(move || run(sink, device, rx, ready_tx))
    .map_err(|e| format!("启动采集线程失败：{e}"))?;

  match ready_rx.recv_timeout(Duration::from_secs(5)) {
    Ok(Ok((device, rate))) => Ok(RecorderHandle { tx, device, rate }),
    Ok(Err(e)) => Err(e),
    Err(_) => Err("打开麦克风超时".into()),
  }
}

fn run(
  sink: Sink,
  device: Option<String>,
  rx: Receiver<Cmd>,
  ready: Sender<Result<(String, u32), String>>,
) {
  let host = cpal::default_host();
  let dev = match pick_device(&host, device.as_deref()) {
    Ok(d) => d,
    Err(e) => {
      let _ = ready.send(Err(e));
      return;
    }
  };
  let name = dev_name(&dev);
  let supported = match dev.default_input_config() {
    Ok(c) => c,
    Err(e) => {
      let _ = ready.send(Err(format!("读取麦克风默认配置失败：{e}")));
      return;
    }
  };
  let rate = supported.sample_rate();
  let channels = supported.channels() as usize;
  let shared = Arc::new(Mutex::new(Shared {
    pre: VecDeque::with_capacity((rate as f32 * PRE_BUF_SEC) as usize + 4096),
    pre_cap: (rate as f32 * PRE_BUF_SEC) as usize,
    active: false,
    paste_mode: PasteMode::CtrlV,
    buf: Vec::with_capacity(rate as usize * 60),
    finalize_at: None,
  }));

  let stream = match build_stream(&dev, &supported, shared.clone(), channels) {
    Ok(s) => s,
    Err(e) => {
      let _ = ready.send(Err(e));
      return;
    }
  };
  if let Err(e) = stream.play() {
    let _ = ready.send(Err(format!("启动音频流失败：{e}")));
    return;
  }
  let _ = ready.send(Ok((name.clone(), rate)));
  crate::log_line(&format!("录音已启动：{name} @ {rate} Hz / {channels} ch"));

  let mut last_emit = Instant::now();
  loop {
    let cmd = rx.recv_timeout(Duration::from_millis(20));
    match cmd {
      Ok(Cmd::Press(mode)) => {
        let mut s = shared.lock().unwrap();
        s.paste_mode = mode;
        s.finalize_at = None;
        if !s.active {
          s.active = true;
          s.buf.clear();
          let pre: Vec<f32> = s.pre.drain(..).collect();
          s.buf.extend_from_slice(&pre);
          drop(s);
          sink.emit_record(true, 0.0);
        }
      }
      Ok(Cmd::Release) => {
        let mut s = shared.lock().unwrap();
        if s.active {
          s.active = false;
          s.finalize_at = Some(Instant::now() + DEBOUNCE);
        }
      }
      Ok(Cmd::Quit) => break,
      Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
      Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
    }

    // 收尾判定
    let ready_samples = {
      let mut s = shared.lock().unwrap();
      let due = s
        .finalize_at
        .map(|t| Instant::now() >= t)
        .unwrap_or(false)
        && !s.active;
      if due {
        s.finalize_at = None;
        // 松开后的尾音仍在预缓冲里，接在本段后面
        let tail: Vec<f32> = s.pre.drain(..).collect();
        s.buf.extend_from_slice(&tail);
        let samples = std::mem::take(&mut s.buf);
        s.buf = Vec::with_capacity(rate as usize * 60);
        Some((samples, s.paste_mode))
      } else {
        None
      }
    };

    if let Some((mut samples, paste_mode)) = ready_samples {
      let dur = samples.len() as f64 / rate as f64;
      if dur >= MIN_SAMPLES_SEC as f64 {
        mel::agc(&mut samples);
        sink.finish_record(
          dur,
          Some(Job { samples, rate, duration_s: dur, paste_mode }),
        );
      } else {
        sink.finish_record(dur, None);
        crate::log_line(&format!("录音过短（{dur:.2}s），忽略"));
      }
    }

    if last_emit.elapsed() > Duration::from_millis(100) {
      last_emit = Instant::now();
    }
  }
  drop(stream);
  crate::log_line("录音线程退出");
}

fn pick_device(host: &cpal::Host, wanted: Option<&str>) -> Result<cpal::Device, String> {
  if let Some(w) = wanted.filter(|w| !w.trim().is_empty()) {
    if let Ok(devices) = host.input_devices() {
      for d in devices {
        if dev_name(&d) == w {
          return Ok(d);
        }
      }
    }
    crate::log_line(&format!("找不到麦克风「{w}」，改用系统默认设备"));
  }
  host
    .default_input_device()
    .ok_or_else(|| "没有可用的麦克风设备".to_string())
}

fn build_stream(
  device: &cpal::Device,
  supported: &cpal::SupportedStreamConfig,
  shared: Arc<Mutex<Shared>>,
  channels: usize,
) -> Result<cpal::Stream, String> {
  let err_fn = |e| crate::log_line(&format!("音频流错误：{e}"));
  let config: cpal::StreamConfig = supported.clone().into();
  let mut sink = move |data: &[f32]| push_shared(&shared, data, channels);

  use cpal::SampleFormat as F;
  let stream = match supported.sample_format() {
    F::F32 => device.build_input_stream(
      config.clone(),
      move |data: &[f32], _| sink(data),
      err_fn,
      None,
    ),
    F::I16 => {
      let mut f = move |data: &[i16]| {
        let v: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
        sink(&v);
      };
      device.build_input_stream(config.clone(), move |data: &[i16], _| f(data), err_fn, None)
    }
    F::U16 => {
      let mut f = move |data: &[u16]| {
        let v: Vec<f32> = data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).collect();
        sink(&v);
      };
      device.build_input_stream(config.clone(), move |data: &[u16], _| f(data), err_fn, None)
    }
    F::I32 => {
      let mut f = move |data: &[i32]| {
        let v: Vec<f32> = data.iter().map(|&s| s as f32 / 2147483648.0).collect();
        sink(&v);
      };
      device.build_input_stream(config.clone(), move |data: &[i32], _| f(data), err_fn, None)
    }
    other => return Err(format!("不支持的采样格式：{other:?}")),
  };
  stream.map_err(|e| format!("创建音频流失败：{e}"))
}

fn dev_name(device: &cpal::Device) -> String {
  device.description().map(|d| d.name().to_string()).unwrap_or_default()
}

fn push_shared(shared: &Arc<Mutex<Shared>>, data: &[f32], channels: usize) {
  let mono = mel::to_mono(data, channels);
  if let Ok(mut s) = shared.try_lock() {
    if s.active {
      if s.buf.len() + mono.len() <= s.pre_cap.max(1) * 120 {
        s.buf.extend_from_slice(&mono);
      }
    } else {
      s.pre.extend(mono);
      while s.pre.len() > s.pre_cap {
        s.pre.pop_front();
      }
    }
  }
}
