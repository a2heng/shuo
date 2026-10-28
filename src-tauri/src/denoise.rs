// 说 · Shuo — 音频降噪（NFT_960 流式 ONNX，48 kHz）
//
// 对应 Python 版 `onnx_infer/denoise.py`：sqrt-Hann STFT → 逐帧 ONNX（带三组状态缓存）
// → 重叠相加 ISTFT。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

pub const N_FFT: usize = 960;
pub const HOP: usize = 480;
pub const SR: usize = 48000;
pub const BINS: usize = N_FFT / 2 + 1;

pub struct Denoiser {
  session: Session,
  conv: Vec<f32>,
  tfa: Vec<f32>,
  inter: Vec<f32>,
  window: Vec<f32>,
}

impl Denoiser {
  pub fn load(path: &Path) -> Result<Self, String> {
    let session = Session::builder()
      .map_err(|e| e.to_string())?
      .with_intra_threads(1)
      .map_err(|e| e.to_string())?
      .with_inter_threads(1)
      .map_err(|e| e.to_string())?
      .with_intra_op_spinning(false)
      .map_err(|e| e.to_string())?
      .commit_from_file(path)
      .map_err(|e| format!("加载降噪模型 {} 失败：{e}", path.display()))?;
    let window = (0..N_FFT)
      .map(|i| {
        let h = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (N_FFT as f64 - 1.0)).cos());
        (h.max(0.0)).sqrt() as f32
      })
      .collect();
    Ok(Self {
      session,
      conv: vec![0.0; 8808],
      tfa: vec![0.0; 610],
      inter: vec![0.0; 4608],
      window,
    })
  }

  pub fn reset(&mut self) {
    self.conv.fill(0.0);
    self.tfa.fill(0.0);
    self.inter.fill(0.0);
  }

  pub fn denoise(&mut self, audio: &[f32]) -> Result<Vec<f32>, String> {
    if audio.len() < N_FFT {
      return Ok(audio.to_vec());
    }
    self.reset();
    let length = audio.len();
    let n_frames = 1 + (length - N_FFT) / HOP;

    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(N_FFT);
    let inv = planner.plan_fft_inverse(N_FFT);
    let mut buf = vec![Complex32::default(); N_FFT];

    let mut out = vec![0.0f32; length];
    let mut wsum = vec![0.0f32; length];

    for f in 0..n_frames {
      let start = f * HOP;
      for k in 0..N_FFT {
        buf[k] = Complex32::new(audio[start + k] * self.window[k], 0.0);
      }
      fwd.process(&mut buf);

      let mut spec = Vec::with_capacity(BINS * 2);
      for k in 0..BINS {
        spec.push(buf[k].re);
        spec.push(buf[k].im);
      }
      let spec_t = Tensor::from_array(([1i64, BINS as i64, 1, 2], spec))
        .map_err(|e| format!("构造 spec 失败：{e}"))?;
      let conv_t = Tensor::from_array(([1i64, 8808], std::mem::take(&mut self.conv)))
        .map_err(|e| format!("构造 conv_cache 失败：{e}"))?;
      let tfa_t = Tensor::from_array(([1i64, 610], std::mem::take(&mut self.tfa)))
        .map_err(|e| format!("构造 tfa_cache 失败：{e}"))?;
      let inter_t = Tensor::from_array(([1i64, 4608], std::mem::take(&mut self.inter)))
        .map_err(|e| format!("构造 inter_cache 失败：{e}"))?;

      let outputs = self
        .session
        .run(ort::inputs![
          "spec" => spec_t,
          "conv_cache" => conv_t,
          "tfa_cache" => tfa_t,
          "inter_cache" => inter_t
        ])
        .map_err(|e| {
          self.conv = vec![0.0; 8808];
          self.tfa = vec![0.0; 610];
          self.inter = vec![0.0; 4608];
          format!("降噪推理失败：{e}")
        })?;

      let (_, enh) = outputs["enhanced_spec"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取 enhanced_spec 失败：{e}"))?;
      let (_, conv) = outputs["conv_cache_out"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取 conv_cache_out 失败：{e}"))?;
      let (_, tfa) = outputs["tfa_cache_out"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取 tfa_cache_out 失败：{e}"))?;
      let (_, inter) = outputs["inter_cache_out"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取 inter_cache_out 失败：{e}"))?;
      self.conv.clear();
      self.conv.extend_from_slice(conv);
      self.tfa.clear();
      self.tfa.extend_from_slice(tfa);
      self.inter.clear();
      self.inter.extend_from_slice(inter);

      for k in 0..N_FFT {
        buf[k] = Complex32::new(0.0, 0.0);
      }
      for k in 0..BINS {
        buf[k] = Complex32::new(enh[k * 2], enh[k * 2 + 1]);
      }
      // irfft：补共轭对称
      for k in 1..BINS - 1 {
        buf[N_FFT - k] = buf[k].conj();
      }
      inv.process(&mut buf);

      let end = (start + N_FFT).min(length);
      for i in start..end {
        let v = buf[i - start].re / N_FFT as f32 * self.window[i - start];
        out[i] += v;
        wsum[i] += self.window[i - start] * self.window[i - start];
      }
    }

    for i in 0..length {
      if wsum[i] > 1e-10 {
        out[i] /= wsum[i];
      }
    }
    Ok(out)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::time::Instant;

  /// 降噪耗时基准（手动跑）：
  /// `SHUO_GGUF_WAV=… cargo test --release denoise_bench -- --ignored --nocapture`
  #[test]
  #[ignore = "bench: 需要 resources/denoise.onnx 与 wav"]
  fn denoise_bench() {
    let model = std::env::var("SHUO_DENOISE_ONNX").unwrap_or_else(|_| {
      format!("{}/resources/denoise.onnx", env!("CARGO_MANIFEST_DIR"))
    });
    let Ok(wav_path) = std::env::var("SHUO_GGUF_WAV") else {
      eprintln!("跳过：SHUO_GGUF_WAV 未设置");
      return;
    };
    let mut reader =
      crate::asr::hound_lite::Wav::open(Path::new(&wav_path)).expect("wav");
    let rate = reader.rate();
    let wav16 = reader.read_i16_mono();
    let mut wav: Vec<f32> = wav16.iter().map(|&s| s as f32 / 32768.0).collect();
    if rate != SR as u32 {
      wav = crate::dsp::resample(&wav, rate, SR as u32).expect("resample");
    }
    let audio_s = wav.len() as f64 / SR as f64;
    let t = Instant::now();
    let mut d = Denoiser::load(Path::new(&model)).expect("load");
    eprintln!("LOAD {:.2}s", t.elapsed().as_secs_f64());
    for i in 0..3 {
      let t = Instant::now();
      let out = d.denoise(&wav).expect("denoise");
      let wall = t.elapsed().as_secs_f64();
      eprintln!(
        "RUN{i} audio={audio_s:.2}s denoise={wall:.2}s RTF={:.3} samples={}",
        wall / audio_s,
        out.len()
      );
      d.reset();
    }
  }
}
