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

/// 电平摘要（日志用）：`峰值 -1.9 dBFS / RMS -20.3 dBFS`。
pub fn levels(samples: &[f32]) -> String {
  if samples.is_empty() {
    return "空".into();
  }
  let peak = samples.iter().fold(0.0f32, |a, v| a.max(v.abs()));
  let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
  let db = |x: f32| 20.0 * x.max(1e-9).log10();
  format!("峰值 {:.1} dBFS / RMS {:.1} dBFS", db(peak), db(rms))
}

/// 唯一的电平处理环节：**压缩器**（低抬、高降，最后整段对齐到峰值 0.8）。
///
/// 之前是「AGC 整段等比缩放」，只能把一句最响的地方拉到目标，句子内部的强弱差原样保留
/// （小声的字还是小声，爆音还是爆音）。现在全交给压缩器，三步：
///
///   ① **逐帧压缩**（20 ms 帧）：每帧 RMS 拉向 -20 dBFS，低抬最多 +30 dB、高降最多 -18 dB；
///      增益包络按 **快抬 5 ms / 慢降 120 ms** 平滑，避免抽气；停顿（比语音电平低 30 dB
///      以上的帧）不动，否则词间底噪被抬起来像喘气。
///   ② **整段对齐**：处理后把 99 分位峰值整体拉到 0.8（-1.9 dBFS），所以不管麦克风多小声、
///      说话多大声，送进模型的电平都落在同一个位置（make-up 增益封顶 +40 dB）。
///   ③ **软限幅**：0.8 以上的尖峰用 tanh 渐进压到 0.98 以内，不是硬切，不打齿。
///
/// 整段帧 RMS 都低于 -62 dBFS（基本只有底噪）时原样返回，避免把噪声放大成"语音"。
pub fn compress(samples: &mut [f32], rate: u32) {
  const TARGET_RMS_DB: f32 = -20.0;
  const GATE_BELOW_SPEECH_DB: f32 = 30.0; // 比语音电平低 30 dB 以上算停顿
  const NO_SPEECH_DB: f32 = -62.0; // 帧 RMS 都没到这个电平 → 当作没说话（仅底噪）
  const MAX_BOOST_DB: f32 = 30.0;
  const MAX_CUT_DB: f32 = 18.0;
  const ATTACK_S: f32 = 0.005;
  const RELEASE_S: f32 = 0.120;
  const MAKEUP_MAX_DB: f32 = 40.0; // 整段对齐时最多再抬这么多
  const TARGET_PEAK: f32 = 0.8; // -1.9 dBFS
  const KNEE: f32 = 0.80;
  const CEILING: f32 = 0.98;

  if samples.is_empty() || rate == 0 {
    return;
  }
  let db = |x: f32| 20.0 * x.max(1e-9).log10();
  let frame = ((rate as f32 * 0.020) as usize).max(1);
  let n = samples.len();

  // ① 逐帧电平与目标增益
  let levels: Vec<f32> = (0..n.div_ceil(frame))
    .map(|f| {
      let s = f * frame;
      let e = (s + frame).min(n);
      let rms = (samples[s..e].iter().map(|v| v * v).sum::<f32>() / (e - s) as f32).sqrt();
      db(rms)
    })
    .collect();
  let mut sorted = levels.clone();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let percentile = |q: f32| sorted[((sorted.len() as f64 - 1.0) * q as f64).round() as usize];
  let speech_db = percentile(0.95);
  if speech_db < NO_SPEECH_DB {
    return; // 整段都只是底噪：不碰
  }
  let gate_db = speech_db - GATE_BELOW_SPEECH_DB;

  let gains: Vec<f32> = levels
    .iter()
    .map(|&level| {
      if level < gate_db {
        0.0
      } else {
        (TARGET_RMS_DB - level).clamp(-MAX_CUT_DB, MAX_BOOST_DB)
      }
    })
    .collect();

  // 以样本为单位的增益包络：变大用 attack，变小用 release
  let a_up = (-1.0 / (ATTACK_S * rate as f32)).exp();
  let a_down = (-1.0 / (RELEASE_S * rate as f32)).exp();
  let mut env_db = gains[0];
  for (i, v) in samples.iter_mut().enumerate() {
    let target = gains[i / frame];
    let a = if target > env_db { a_up } else { a_down };
    env_db = a * env_db + (1.0 - a) * target;
    *v *= 10f32.powf(env_db / 20.0);
  }

  // ② 整段对齐：99 分位峰值 → 0.8（-1.9 dBFS）
  let mut mags: Vec<f32> = samples.iter().map(|v| v.abs()).collect();
  mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let peak = mags[((mags.len() as f64 - 1.0) * 0.99).round() as usize];
  if peak >= 1e-5 {
    let makeup = (TARGET_PEAK / peak).clamp(0.5, 10f32.powf(MAKEUP_MAX_DB / 20.0));
    for v in samples.iter_mut() {
      *v *= makeup;
    }
  }

  // ③ 软限幅：0.8 以上渐进压缩，渐近上限 0.98
  for v in samples.iter_mut() {
    let a = v.abs();
    if a > KNEE {
      let shaped = KNEE + (CEILING - KNEE) * ((a - KNEE) / (CEILING - KNEE)).tanh();
      *v = v.signum() * shaped;
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

  /// 压缩器设计验证（合成波形实验，纯 CPU 毫秒级）。
  ///
  /// 设计目标：
  ///   ① 任何音量（-50…0 dBFS）→ 处理后 99 分位峰值都落在 0.8（-1.9 dBFS）附近；
  ///   ② 一句话内部四段强弱悬殊（32 dB）→ 压到 10 dB 以内；
  ///   ③ 停顿/底噪（比语音低 30 dB 以上）不被抬起来；
  ///   ④ 只有底噪（帧 RMS 全程低于 -62 dBFS）→ 原样返回；
  ///   ⑤ 单点爆音不把语音压小，也不削波（软限幅渐近 0.98）。
  ///
  /// `cargo test --release compress_design -- --nocapture` 看表格；
  /// 波形写到 `/tmp/opencode/agc/*.f32`，配 `docs/agc-levels.py` 出图。
  #[test]
  fn compress_design_experiment() {
    const OUT_DIR: &str = "/tmp/opencode/agc";
    const SR: u32 = 48000;
    fn db(x: f32) -> f32 {
      20.0 * x.max(1e-9).log10()
    }
    fn peak(x: &[f32]) -> f32 {
      x.iter().fold(0.0f32, |a, v| a.max(v.abs()))
    }
    fn rms(x: &[f32]) -> f32 {
      (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }
    fn dump(name: &str, before: &[f32], after: &[f32]) {
      let _ = std::fs::create_dir_all(OUT_DIR);
      let bytes = |v: &[f32]| v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
      let _ = std::fs::write(format!("{OUT_DIR}/{name}.in.f32"), bytes(before));
      let _ = std::fs::write(format!("{OUT_DIR}/{name}.out.f32"), bytes(after));
    }
    /// 类语音信号：基频 + 谐波，整体缩放到目标峰值。
    fn speech_like(target_peak: f32, secs: f32) -> Vec<f32> {
      let n = (SR as f32 * secs) as usize;
      let mut v: Vec<f32> = (0..n)
        .map(|i| {
          let t = i as f32 / SR as f32;
          let f0 = 180.0;
          (0..6)
            .map(|k| (2.0 * std::f32::consts::PI * f0 * (k as f32 + 1.0) * t).sin() / (k as f32 + 1.5))
            .sum::<f32>()
        })
        .collect();
      let g = target_peak / peak(&v);
      v.iter_mut().for_each(|s| *s *= g);
      v
    }

    eprintln!("\n压缩器实验：逐帧压缩（目标 RMS -20 dBFS）→ 整段对齐 0.8 峰值 → 软限幅 0.98");
    eprintln!(
      "{:>9} {:>11} {:>11} {:>10} {:>8}",
      "输入峰值", "输出峰值", "输出RMS", "总增益", "结论"
    );
    // -38 dBFS ≈ 本机麦克风正常说话的实际原始峰值
    for &(name, target_db) in &[
      ("loud", -3.0f32),
      ("normal", -18.0),
      ("soft", -30.0),
      ("mic", -38.0),
      ("quiet", -44.0),
      ("whisper", -50.0),
      ("silence", -62.0),
      ("noise", -75.0),
    ] {
      let before = speech_like(10f32.powf(target_db / 20.0), 1.0);
      let mut after = before.clone();
      compress(&mut after, SR);
      dump(name, &before, &after);
      let g = 20.0 * (peak(&after) / peak(&before)).log10();
      let untouched = after == before;
      eprintln!(
        "{:>8.1} dB {:>8.1} dB {:>8.1} dB {:>7.1} dB  {}",
        db(peak(&before)),
        db(peak(&after)),
        db(rms(&after)),
        g,
        if untouched {
          "原样返回（没说话）"
        } else {
          "对齐到 -1.9 dBFS"
        }
      );
      assert!(peak(&after) <= 0.981, "{name}：不允许软限幅溢出 0.98");
      if db(rms(&before)) <= -62.0 {
        assert!(untouched, "{name}：整段只有底噪时必须原样返回");
      } else {
        assert!(
          (db(peak(&after)) - db(0.8)).abs() < 1.0,
          "{name}：输出峰值应 ≈ -1.9 dBFS，实际 {:.1} dBFS",
          db(peak(&after))
        );
      }
    }

    // ② 段内动态：四段强弱悬殊的"一句话"（真实说话音量波动大致就在 20 dB 以内）
    let seg_db = [-30.0f32, -22.0, -8.0, -26.0];
    let mut raw: Vec<f32> = Vec::new();
    for &lvl in &seg_db {
      raw.extend(speech_like(10f32.powf(lvl / 20.0), 0.4));
    }
    let mut after = raw.clone();
    compress(&mut after, SR);
    dump("compress", &raw, &after);
    let per = after.len() / seg_db.len();
    let seg_rms = |v: &[f32], i: usize| db(rms(&v[i * per..(i + 1) * per]));
    eprintln!("\n一句话内四段强弱（每段 0.4 s）：");
    eprintln!("{:>4} {:>10} {:>10} {:>9}", "段", "处理前RMS", "处理后RMS", "变化");
    let (mut lo_b, mut hi_b, mut lo_a, mut hi_a) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for i in 0..seg_db.len() {
      let (b, a) = (seg_rms(&raw, i), seg_rms(&after, i));
      lo_b = lo_b.min(b);
      hi_b = hi_b.max(b);
      lo_a = lo_a.min(a);
      hi_a = hi_a.max(a);
      eprintln!("{:>4} {:>7.1} dB {:>7.1} dB {:>6.1} dB", i + 1, b, a, a - b);
    }
    eprintln!(
      "段间动态：{:.1} dB → {:.1} dB；峰值 {:.1} dBFS\n",
      hi_b - lo_b,
      hi_a - lo_a,
      db(peak(&after))
    );
    assert!(
      hi_a - lo_a < (hi_b - lo_b) * 0.5,
      "压缩后段间动态应至少减半（{:.1} → {:.1} dB）",
      hi_b - lo_b,
      hi_a - lo_a
    );

    // ③ 停顿：说话 → 底噪 → 说话，停顿段不许被抬
    let mut with_gap = speech_like(0.2, 0.5);
    let gap_start = with_gap.len();
    with_gap.extend(speech_like(10f32.powf(-55.0 / 20.0), 0.5));
    let gap_end = with_gap.len();
    with_gap.extend(speech_like(0.2, 0.5));
    let mut gapped = with_gap.clone();
    compress(&mut gapped, SR);
    let (gb, ga) = (db(rms(&with_gap[gap_start..gap_end])), db(rms(&gapped[gap_start..gap_end])));
    eprintln!(
      "停顿段（底噪 -55 dBFS）：{gb:.1} dB → {ga:.1} dB（只跟整段 make-up 平移，不被压到语音电平）"
    );
    // 停顿帧本身不压缩；整段 make-up 是对所有样本的同一个增益，所以停顿只会被动平移
    assert!(
      ga - gb < 15.0,
      "停顿/底噪不应被压到语音电平（{gb:.1} → {ga:.1} dB，抬了 {:.1} dB）",
      ga - gb
    );

    // ④ 瞬态：1 % 采样是爆音
    let mut spike = speech_like(0.2, 1.0);
    let step = (spike.len() / 100).max(1);
    for i in (0..spike.len()).step_by(step) {
      spike[i] = 1.0;
    }
    let mut spiked = spike.clone();
    compress(&mut spiked, SR);
    dump("transient", &spike, &spiked);
    let speech_rms_after = db(rms(&spiked));
    let clipped = spiked.iter().filter(|v| v.abs() >= 0.979).count();
    eprintln!(
      "含 1% 爆音：峰值 {:.1} → {:.1} dBFS，RMS {:.1} dBFS，触顶 {clipped} 点（软限幅）",
      db(peak(&spike)),
      db(peak(&spiked)),
      speech_rms_after
    );
    assert!(peak(&spiked) <= 0.981, "软限幅后峰值不得超过 0.98");
    assert!(speech_rms_after > -30.0, "爆音不应把语音压成很小（RMS {speech_rms_after:.1} dB）");
    eprintln!("波形已导出到 {OUT_DIR}/\n");
  }
}
