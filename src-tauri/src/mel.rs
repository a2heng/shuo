// 说 · Shuo — 梅尔频谱（Whisper / librosa 兼容）
//
// 纯 Rust 实现，与 Python 版 `onnx_infer/asr.py::compute_mel_spectrogram` 数值一致：
// 反射填充 → 汉宁窗 STFT → 功率谱 → Slaney 归一化梅尔滤波器组 → log10 / 动态范围压缩。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::OnceLock;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
pub const N_BINS: usize = N_FFT / 2 + 1;

// Slaney 梅尔刻度常量（librosa 默认 htk=False）
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
const LOGSTEP: f64 = 0.068_751_777_420_597_27; // ln(6.4) / 27

fn hz_to_mel(f: f64) -> f64 {
  if f < MIN_LOG_HZ {
    f / F_SP
  } else {
    MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
  }
}

fn mel_to_hz(m: f64) -> f64 {
  if m < MIN_LOG_MEL {
    m * F_SP
  } else {
    MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_MEL)).exp()
  }
}

/// Slaney 归一化三角滤波器组，形状 [N_MELS, N_BINS]。
pub fn mel_filters() -> &'static [f32] {
  static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
  FILTERS.get_or_init(|| {
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(fmax);
    let n_points = N_MELS + 2;
    let freqs: Vec<f64> = (0..n_points)
      .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (n_points - 1) as f64))
      .collect();
    let fft_freqs: Vec<f64> = (0..N_BINS)
      .map(|i| fmax * i as f64 / (N_BINS - 1) as f64)
      .collect();

    let mut weights = vec![0.0f32; N_MELS * N_BINS];
    for m in 0..N_MELS {
      let lower_slope = freqs[m + 1] - freqs[m];
      let upper_slope = freqs[m + 2] - freqs[m + 1];
      for k in 0..N_BINS {
        // 升坡（频率高于下沿的部分）与降坡（到上沿的距离），任一为负则置 0
        let lower = (fft_freqs[k] - freqs[m]) / lower_slope;
        let upper = (freqs[m + 2] - fft_freqs[k]) / upper_slope;
        let w = lower.min(upper).max(0.0);
        // Slaney 归一化：三角面积归一到 1
        let enorm = 2.0 / (freqs[m + 2] - freqs[m]);
        weights[m * N_BINS + k] = (w * enorm) as f32;
      }
    }
    weights
  })
}

fn hann_window() -> &'static [f32] {
  static W: OnceLock<Vec<f32>> = OnceLock::new();
  W.get_or_init(|| {
    (0..N_FFT)
      .map(|i| (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (N_FFT - 1) as f64).cos())) as f32)
      .collect()
  })
}

/// 计算对数梅尔频谱，返回 `(数据, 帧数)`，数据按 [N_MELS, 帧数] 行优先排列。
pub fn log_mel(wav: &[f32]) -> (Vec<f32>, usize) {
  let pad = N_FFT / 2;
  let n = wav.len();
  // numpy 的 mode='reflect'：以首尾样本为轴镜像，不重复端点
  let padded_len = n + 2 * pad;
  let sample = |i: isize| -> f32 {
    let mut idx = i - pad as isize;
    if idx < 0 {
      idx = -idx;
    }
    if idx >= n as isize {
      idx = 2 * (n as isize - 1) - idx;
    }
    let idx = idx.clamp(0, n as isize - 1);
    wav[idx as usize]
  };
  let padded: Vec<f32> = (0..padded_len as isize).map(sample).collect();

  let frames = 1 + (padded_len - N_FFT) / HOP;
  let window = hann_window();
  let mut planner = FftPlanner::<f32>::new();
  let fft = planner.plan_fft_forward(N_FFT);
  let mut buf = vec![Complex32::default(); N_FFT];

  let filters = mel_filters();
  let mut out = vec![0.0f32; N_MELS * frames];
  let mut power = vec![0.0f32; N_BINS];

  for t in 0..frames {
    let start = t * HOP;
    for k in 0..N_FFT {
      buf[k] = Complex32::new(padded[start + k] * window[k], 0.0);
    }
    fft.process(&mut buf);
    for k in 0..N_BINS {
      power[k] = buf[k].norm_sqr();
    }
    for m in 0..N_MELS {
      let row = &filters[m * N_BINS..(m + 1) * N_BINS];
      let mut acc = 0.0f32;
      for k in 0..N_BINS {
        acc += row[k] * power[k];
      }
      out[m * frames + t] = acc;
    }
  }

  // log10(max(x, 1e-10)) → 动态范围压缩（相对最大值 -8）→ (x + 4) / 4
  let mut max = f32::NEG_INFINITY;
  for v in out.iter_mut() {
    *v = v.max(1e-10).log10();
    if *v > max {
      max = *v;
    }
  }
  let floor = max - 8.0;
  for v in out.iter_mut() {
    *v = (*v).max(floor);
    *v = (*v + 4.0) / 4.0;
  }

  (out, frames)
}

/// 按静音位置切分长音频，返回分段样本边界（含首尾）。与 Python `find_silence_split_points` 一致。
pub fn silence_splits(wav: &[f32], target_sec: usize) -> Vec<usize> {
  let min_sec = (target_sec / 2) as f32;
  let max_sec = (target_sec as f32 * 1.5) as usize;
  let total = wav.len();
  if total <= max_sec * SAMPLE_RATE {
    return Vec::new();
  }

  const SILENCE_HOP_SEC: f32 = 0.1;
  const SILENCE_THRESHOLD_DB: f32 = -40.0;
  let hop_samples = (SILENCE_HOP_SEC * SAMPLE_RATE as f32) as usize;
  let frame_len = hop_samples * 2;
  if wav.len() < frame_len {
    return Vec::new();
  }
  let n_frames = 1 + (wav.len() - frame_len) / hop_samples;
  let mut rms = vec![0.0f32; n_frames];
  for (i, r) in rms.iter_mut().enumerate() {
    let frame = &wav[i * hop_samples..i * hop_samples + frame_len];
    *r = (frame.iter().map(|v| v * v).sum::<f32>() / frame_len as f32).sqrt();
  }
  let max_rms = rms.iter().cloned().fold(0.0f32, f32::max);
  let is_silent: Vec<bool> = rms
    .iter()
    .map(|r| 20.0 * (r.max(1e-10) / max_rms.max(1e-10)).log10() < SILENCE_THRESHOLD_DB)
    .collect();

  let mut splits = Vec::new();
  let mut cursor = 0usize;
  while cursor + max_sec * SAMPLE_RATE < total {
    let cursor_sec = cursor as f32 / SAMPLE_RATE as f32;
    let search_start_sec = (cursor_sec + min_sec).max(0.0);
    let search_end_sec = cursor_sec + max_sec as f32;
    let target_abs_sec = cursor_sec + target_sec as f32;

    let frame_start = (search_start_sec / SILENCE_HOP_SEC) as usize;
    let frame_end = ((search_end_sec / SILENCE_HOP_SEC) as usize).min(is_silent.len());
    let frame_target = (target_abs_sec / SILENCE_HOP_SEC) as isize;

    let best = (frame_start..frame_end)
      .filter(|&f| is_silent[f])
      .min_by_key(|&f| (f as isize - frame_target).abs());

    let split_sample = match best {
      Some(f) => (f * hop_samples).min(total),
      None => ((target_abs_sec as usize) * SAMPLE_RATE).min(total),
    };
    splits.push(split_sample);
    cursor = split_sample;
  }
  splits
}

/// 把线性交错采样转成单声道（取平均）。
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
  if channels <= 1 {
    return interleaved.to_vec();
  }
  let n = interleaved.len() / channels;
  let mut out = vec![0.0f32; n];
  for (i, s) in out.iter_mut().enumerate() {
    let mut acc = 0.0;
    for c in 0..channels {
      acc += interleaved[i * channels + c];
    }
    *s = acc / channels as f32;
  }
  out
}

/// AGC：99 分位峰值归一化到约 -1.9 dBFS（0.8 满量程），与 Python 版一致。
pub fn agc(samples: &mut [f32]) {
  if samples.is_empty() {
    return;
  }
  let mut mags: Vec<f32> = samples.iter().map(|v| v.abs()).collect();
  mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let idx = ((mags.len() as f64 - 1.0) * 0.99).round() as usize;
  let peak = mags[idx];
  if peak > 0.0 {
    let gain = 0.8 / peak;
    for v in samples.iter_mut() {
      *v = (*v * gain).clamp(-1.0, 1.0);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dump_mel_filters() {
    let f = mel_filters();
    let mut bytes = Vec::with_capacity(f.len() * 4);
    for v in f {
      bytes.extend_from_slice(&v.to_le_bytes());
    }
    // 调试转储目录 CI 上不存在，先建再写（Windows 落到当前盘 \tmp\opencode）
    std::fs::create_dir_all("/tmp/opencode").unwrap();
    std::fs::write("/tmp/opencode/mel_filters_rs.f32", &bytes).unwrap();
  }
}
