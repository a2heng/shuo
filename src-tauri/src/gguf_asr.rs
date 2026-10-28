// 说 · Shuo — llama.cpp/GGUF 推理后端（Qwen3-ASR-1.7B，提示词风格控制主路径）
//
// GGUF 后端（llama.cpp）：模型来自官方 ggml-org/Qwen3-ASR-1.7B-GGUF，或
// Release 微调模型包（*-tuned，提示词可控输出风格）。llama-cpp-sys-2
// **源码静态链接**（openmp + mtmd features），无 DLL/so
// 伴生；mtmd 负责音频编码与拼接推理（llama.cpp 官方 Qwen3-ASR 支持，
// tools/mtmd/models/qwen3a.cpp）。方法论与 API 用法移植自
// Qwen3-subtitle-assistant 的 src/gguf_asr.rs（E1/E2 全链路实测：LM+mmproj 加载、
// system 段 context 通道、原始输出 `language X<asr_text>正文` 解析、
// 静音双模式空输出穿越 Q4_K_M 量化不丢）。
//
// 与 ONNX 路径（asr.rs）保持同一对外接口（transcribe/transcribe_single →
// AsrResult），state.rs 的 worker 对两种后端一视同仁；context（输出风格
// 任务开关）语义一致。采样固定贪心（与 ONNX 路径的确定性解码对齐）。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::Once;
use std::time::Instant;

use llama_cpp_sys_2 as sys;

use crate::asr::AsrResult;
use crate::mel::{self, SAMPLE_RATE};

static BACKEND_INIT: Once = Once::new();

/// llama/ggml/mtmd 原生 C 日志 → shuo 日志桥接。
/// 默认只放行 ERROR（加载期元数据 dump / 后端搜索 / 逐段计时等数十~数百行噪声
/// 不进 ~/.shuo/shuo.log）；设 `SHUO_LLAMA_LOG=1` 放行全量（排障用）。
unsafe extern "C" fn native_log_hook(
  level: sys::ggml_log_level,
  text: *const c_char,
  _user_data: *mut c_void,
) {
  if text.is_null() {
    return;
  }
  let verbose = std::env::var_os("SHUO_LLAMA_LOG").is_some_and(|v| v != "0");
  if level != sys::GGML_LOG_LEVEL_ERROR && !verbose {
    return;
  }
  let s = CStr::from_ptr(text).to_string_lossy();
  for line in s.lines() {
    let line = line.trim_end();
    if line.is_empty() {
      continue;
    }
    crate::log_line(&format!("llama: {line}"));
  }
}

fn to_cstring(p: &Path) -> Result<CString, String> {
  CString::new(p.to_string_lossy().as_bytes().to_vec())
    .map_err(|e| format!("路径含非法字符 {}：{e}", p.display()))
}

fn init_backend() {
  BACKEND_INIT.call_once(|| {
    unsafe {
      // 先接管原生日志（在任何 ggml/llama/mtmd 输出发生之前），否则加载噪声直写 stderr
      sys::llama_log_set(Some(native_log_hook), std::ptr::null_mut());
      sys::ggml_log_set(Some(native_log_hook), std::ptr::null_mut());
      sys::mtmd_helper_log_set(Some(native_log_hook), std::ptr::null_mut());
      sys::llama_backend_init();
    }
    let n = unsafe { sys::ggml_backend_dev_count() };
    let mut names = Vec::new();
    for i in 0..n {
      let dev = unsafe { sys::ggml_backend_dev_get(i) };
      if dev.is_null() {
        continue;
      }
      let name = unsafe { CStr::from_ptr(sys::ggml_backend_dev_name(dev)) }
        .to_string_lossy()
        .into_owned();
      names.push(name);
    }
    crate::log_line(&format!("ggml 后端设备：{}", names.join(", ")));
  });
}

/// GGUF 版 Qwen3-ASR 推理引擎：LM + mmproj 音频编码器 + mtmd 上下文。
pub struct GgufAsr {
  model: *mut sys::llama_model,
  ctx: *mut sys::llama_context,
  mtmd: *mut sys::mtmd_context,
  sample_rate: i32,
  n_batch: i32,
  max_new_tokens: i32,
}

// 引擎在单一 worker 线程内串行使用（与 AsrPipeline 的 Mutex 语义一致）
unsafe impl Send for GgufAsr {}

impl GgufAsr {
  /// 加载 LM GGUF 与 mmproj，建立推理上下文（CPU 静态构建：ngl=0）。
  pub fn load(lm: &Path, mmproj: &Path, threads: usize, max_new_tokens: usize) -> Result<Self, String> {
    init_backend();
    let threads = threads.clamp(1, 16) as i32;
    unsafe {
      let c_lm = to_cstring(lm)?;
      let mut mparams = sys::llama_model_default_params();
      mparams.n_gpu_layers = 0;
      let model = sys::llama_model_load_from_file(c_lm.as_ptr(), mparams);
      if model.is_null() {
        return Err(format!(
          "GGUF 模型加载失败：{}（确认文件完整且为 llama.cpp 格式）",
          lm.display()
        ));
      }

      let mut cparams = sys::llama_context_default_params();
      cparams.n_ctx = 4096; // 音频 12.5 token/s + 文本 prompt + 生成，4k 足够单段
      cparams.n_batch = 2048; // 音频 chunk 一次 prefill 需要较大 batch
      cparams.n_threads = threads;
      cparams.n_threads_batch = threads;
      let ctx = sys::llama_init_from_model(model, cparams);
      if ctx.is_null() {
        sys::llama_model_free(model);
        return Err("llama_context 创建失败".into());
      }

      let c_mm = to_cstring(mmproj)?;
      let mparams2 = sys::mtmd_context_params_default();
      let mtmd = sys::mtmd_init_from_file(c_mm.as_ptr(), model, mparams2);
      if mtmd.is_null() {
        sys::llama_free(ctx);
        sys::llama_model_free(model);
        return Err(format!(
          "mmproj 加载失败：{}（需要与 LM 同一次转换产出）",
          mmproj.display()
        ));
      }
      if !sys::mtmd_support_audio(mtmd) {
        sys::mtmd_free(mtmd);
        sys::llama_free(ctx);
        sys::llama_model_free(model);
        return Err("该 mmproj 不支持音频输入（不是 Qwen3-ASR 的多模态投影？）".into());
      }
      let sample_rate = sys::mtmd_get_audio_sample_rate(mtmd);
      crate::log_line(&format!(
        "GGUF ASR 就绪：LM {} + mmproj {}（目标采样率 {} Hz，线程 {threads}）",
        lm.file_name().unwrap_or_default().to_string_lossy(),
        mmproj.file_name().unwrap_or_default().to_string_lossy(),
        sample_rate,
      ));
      Ok(Self {
        model,
        ctx,
        mtmd,
        sample_rate,
        n_batch: cparams.n_batch as i32,
        max_new_tokens: max_new_tokens.max(16) as i32,
      })
    }
  }

  /// 目标音频采样率（Qwen3-ASR 为 16000；调用方负责重采样）。
  pub fn sample_rate(&self) -> i32 {
    self.sample_rate
  }

  /// LM 权重体积（MB），状态展示用。
  pub fn model_size_mb(&self) -> f64 {
    unsafe { sys::llama_model_size(self.model) as f64 / 1e6 }
  }

  /// 转录一段音频（贪心解码），返回解析后的 (正文, 语言标签, token 数)。
  ///
  /// `context` 进 chat 模板 system 段（输出风格任务开关；空串=原生转写）；
  /// `language` 非空时以 assistant 前缀强制（与 ONNX 路径同语义）。
  /// 推理链与 llama.cpp mtmd-helper / llama-server 同款（qsa E1/E2 已实证）：
  /// 清 KV → 模板拼 prompt（音频位用 mtmd marker）→ mtmd_tokenize(text+audio)
  /// → mtmd_helper_eval_chunks（音频编码+文本 prefill 一次完成）→ 贪心到 EOG
  /// → detokenize → 解析 `language X<asr_text>正文`。
  fn transcribe_chunk(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
  ) -> Result<(String, String, usize), String> {
    if wav.is_empty() {
      return Err("音频为空".into());
    }
    unsafe {
      // 0) 清空 KV cache（每段独立会话；-1,-1,-1 = 全部序列全部位置）
      let mem = sys::llama_get_memory(self.ctx);
      sys::llama_memory_seq_rm(mem, -1, -1, -1);

      // 1) prompt：Qwen3-ASR 官方 chat 模板（与 asr.rs prompt_ids / 模型训练
      //    prefix 逐 token 同构），音频占位用 mtmd marker
      let marker = CStr::from_ptr(sys::mtmd_get_marker(self.mtmd))
        .to_string_lossy()
        .into_owned();
      let ctx_seg = context.trim();
      let mut prompt = format!(
        "<|im_start|>system\n{ctx_seg}<|im_end|>\n<|im_start|>user\n<|fim_prefix|>{marker}<|vision_end|><|im_end|>\n<|im_start|>assistant\n"
      );
      if let Some(lang) = language {
        prompt.push_str(&format!("language {lang}<asr_text>"));
      }
      let c_prompt =
        CString::new(prompt).map_err(|e| format!("prompt 含非法字符：{e}"))?;

      // 2) 音频 bitmap（16k f32 单声道）
      let bitmap = sys::mtmd_bitmap_init_from_audio(wav.len(), wav.as_ptr());
      if bitmap.is_null() {
        return Err("mtmd_bitmap_init_from_audio 失败".into());
      }

      // 3) tokenize：文本按 marker 切开，音频 chunk 插到 marker 位
      let chunks = sys::mtmd_input_chunks_init();
      if chunks.is_null() {
        sys::mtmd_bitmap_free(bitmap);
        return Err("mtmd_input_chunks_init 失败".into());
      }
      let text = sys::mtmd_input_text {
        text: c_prompt.as_ptr(),
        text_len: c_prompt.as_bytes().len(),
        add_special: false,  // 模板已含全部特殊 token 字面量
        parse_special: true, // <|im_start|> 等按特殊 token 解析
      };
      let bitmaps: [*const sys::mtmd_bitmap; 1] = [bitmap];
      let rc = sys::mtmd_tokenize(self.mtmd, chunks, &text, bitmaps.as_ptr(), 1);
      if rc != 0 {
        sys::mtmd_input_chunks_free(chunks);
        sys::mtmd_bitmap_free(bitmap);
        return Err(format!("mtmd_tokenize 失败 (rc={rc})"));
      }

      // 4) prefill：文本 chunk 走 llama_decode，音频 chunk 走 mtmd 编码后拼接
      let mut n_past: sys::llama_pos = 0;
      let rc = sys::mtmd_helper_eval_chunks(
        self.mtmd,
        self.ctx,
        chunks,
        0, // n_past 起点
        0, // seq_id
        self.n_batch,
        true, // logits_last：末位出 logits 供采样
        &mut n_past,
      );
      sys::mtmd_input_chunks_free(chunks);
      sys::mtmd_bitmap_free(bitmap);
      if rc != 0 {
        return Err(format!("mtmd_helper_eval_chunks 失败 (rc={rc})"));
      }

      // 5) 贪心解码到 EOG（与 ONNX 路径的 argmax 语义一致，确定性输出）
      let vocab = sys::llama_model_get_vocab(self.model);
      let smpl = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
      sys::llama_sampler_chain_add(smpl, sys::llama_sampler_init_greedy());

      let mut out = Vec::<u8>::new();
      let mut piece = vec![0i8; 1024];
      let mut n_tok = 0;
      while n_tok < self.max_new_tokens {
        let mut tok = sys::llama_sampler_sample(smpl, self.ctx, -1);
        if sys::llama_vocab_is_eog(vocab, tok) {
          break;
        }
        // pin 版本（=0.1.157）签名：(vocab, token, buf, length, lstrip, special)
        let n = sys::llama_token_to_piece(
          vocab,
          tok,
          piece.as_mut_ptr() as *mut c_char,
          piece.len() as c_int,
          0,
          true,
        );
        if n > 0 {
          out.extend_from_slice(std::slice::from_raw_parts(piece.as_ptr() as *const u8, n as usize));
        }
        let batch = sys::llama_batch_get_one(&mut tok as *mut sys::llama_token, 1);
        if sys::llama_decode(self.ctx, batch) != 0 {
          break;
        }
        n_tok += 1;
      }
      sys::llama_sampler_free(smpl);
      let raw = String::from_utf8_lossy(&out).into_owned();

      // 6) 解析（与 asr.rs transcribe_chunk 同语义）：
      //    `language X<asr_text>正文`；强制 language 前缀时正文裸出
      let mut lang = String::new();
      let mut text = raw.clone();
      if let Some((lang_part, rest)) = raw.split_once("<asr_text>") {
        if let Some(l) = lang_part.trim().strip_prefix("language ") {
          lang = l.trim().to_string();
        }
        text = rest.to_string();
      } else if let Some(l) = language {
        lang = l.to_string();
      }
      Ok((text.trim().to_string(), lang, n_tok as usize))
    }
  }

  /// 转写一段 16 kHz 单声道音频；长音频按静音切分（与 AsrPipeline::transcribe 同接口）。
  pub fn transcribe(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    _max_new_tokens: usize,
    chunk_sec: usize,
  ) -> Result<AsrResult, String> {
    let t0 = Instant::now();
    let audio_s = wav.len() as f64 / SAMPLE_RATE as f64;
    let splits = mel::silence_splits(wav, chunk_sec);
    if splits.is_empty() {
      let (text, lang, tokens) = self.transcribe_chunk(wav, language, context)?;
      return Ok(AsrResult {
        text,
        language: lang,
        audio_s,
        asr_s: t0.elapsed().as_secs_f64(),
        tokens,
        chunks: 1,
      });
    }
    let mut bounds = vec![0usize];
    bounds.extend(splits.iter().copied());
    bounds.push(wav.len());
    let chunks = bounds.len() - 1;
    let mut texts = Vec::with_capacity(chunks);
    let mut lang = language.unwrap_or("").to_string();
    let mut tokens = 0usize;
    for i in 0..chunks {
      let (text, l, n) = self.transcribe_chunk(&wav[bounds[i]..bounds[i + 1]], language, context)?;
      let text = text.trim();
      if !text.is_empty() {
        texts.push(text.to_string());
      }
      if lang.is_empty() {
        lang = l;
      }
      tokens += n;
    }
    Ok(AsrResult {
      text: texts.join(" "),
      language: lang,
      audio_s,
      asr_s: t0.elapsed().as_secs_f64(),
      tokens,
      chunks,
    })
  }

  /// 无分段（降噪路径使用，与 AsrPipeline::transcribe_single 同接口）。
  pub fn transcribe_single(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    _max_new_tokens: usize,
  ) -> Result<AsrResult, String> {
    let t0 = Instant::now();
    let (text, lang, tokens) = self.transcribe_chunk(wav, language, context)?;
    Ok(AsrResult {
      text,
      language: lang,
      audio_s: wav.len() as f64 / SAMPLE_RATE as f64,
      asr_s: t0.elapsed().as_secs_f64(),
      tokens,
      chunks: 1,
    })
  }
}

impl Drop for GgufAsr {
  fn drop(&mut self) {
    unsafe {
      sys::mtmd_free(self.mtmd);
      sys::llama_free(self.ctx);
      sys::llama_model_free(self.model);
    }
  }
}

/// 解析 Qwen3-ASR 原始输出 `language X<asr_text>正文`（llama.cpp 不做解析，返回原始输出）。
pub fn parse_asr_output(raw: &str) -> (String, String) {
  let s = raw.trim();
  match s.split_once("<asr_text>") {
    Some((head, body)) => {
      let lang = head
        .trim()
        .strip_prefix("language")
        .unwrap_or("")
        .trim()
        .to_string();
      (body.trim().to_string(), lang)
    }
    None => (s.to_string(), String::new()),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_asr_output_variants() {
    assert_eq!(
      parse_asr_output("language Chinese<asr_text>你好。"),
      ("你好。".to_string(), "Chinese".to_string())
    );
    assert_eq!(parse_asr_output("language None<asr_text>"), (String::new(), "None".to_string()));
    assert_eq!(
      parse_asr_output("language English<asr_text> Hello world "),
      ("Hello world".to_string(), "English".to_string())
    );
    assert_eq!(parse_asr_output("裸文本无标签"), ("裸文本无标签".to_string(), String::new()));
    // lang=None 但带正文（实测存在）：正文保留，交由调用方处理
    assert_eq!(parse_asr_output("language None<asr_text>今天"), ("今天".to_string(), "None".to_string()));
  }

  /// 真实模型冒烟（仅 SHUO_GGUF_LM / SHUO_GGUF_MMPROJ 存在时跑，
  /// CI 的 gguf-e2e 模式使用）：加载 → 转录参考 wav → 断言非空 + context 冒烟。
  #[test]
  fn gguf_transcribe_reference_wav() {
    let (Ok(lm), Ok(mm)) = (std::env::var("SHUO_GGUF_LM"), std::env::var("SHUO_GGUF_MM")) else {
      eprintln!("跳过：SHUO_GGUF_LM / SHUO_GGUF_MM 未设置");
      return;
    };
    let wav_path = std::env::var("SHUO_GGUF_WAV").unwrap_or_else(|_| {
      "models/Qwen3-ASR-0.6B-ONNX-CPU/test_audio/librispeech_1_1089_1.wav".to_string()
    });
    let mut reader = crate::asr::hound_lite::Wav::open(std::path::Path::new(&wav_path)).expect("wav");
    let rate = reader.rate();
    let wav16 = reader.read_i16_mono();
    let mut wav: Vec<f32> = wav16.iter().map(|&s| s as f32 / 32768.0).collect();
    if rate != SAMPLE_RATE as u32 {
      wav = crate::dsp::resample(&wav, rate, SAMPLE_RATE as u32).expect("resample");
    }
    let mut engine =
      GgufAsr::load(std::path::Path::new(&lm), std::path::Path::new(&mm), 4, 512).expect("load");
    assert_eq!(engine.sample_rate(), SAMPLE_RATE as i32);
    let r = engine.transcribe(&wav, None, "", 512, 30).expect("transcribe");
    eprintln!("TEXT={} LANG={} asr_s={:.2}", r.text, r.language, r.asr_s);
    assert!(!r.text.trim().is_empty(), "转写为空");
    // context 冒烟：基座带任务提示词也必须正常出文本（指令执行由微调权重兑现）
    let r2 = engine
      .transcribe(&wav, None, "translate to English", 512, 30)
      .expect("ctx transcribe");
    eprintln!("CTX TEXT={}", r2.text);
    assert!(!r2.text.trim().is_empty(), "context 模式转写为空");
  }
}
