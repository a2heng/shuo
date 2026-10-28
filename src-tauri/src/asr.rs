// 说 · Shuo — Qwen3-ASR-0.6B 纯 ONNX 推理管线（Rust 端口）
//
// 对应 Python 版 `onnx_infer/asr.py`：梅尔谱 → encoder(conv + transformer) → 提示词拼接
// （音频占位符替换为 encoder 输出）→ decoder_init 预填充 → decoder_step 贪心解码。
//
// 与 Python 版的数值对齐由 `tests` 里的 mel / 文本对照保证；改动本文件请先跑
// `cargo test --release -- --nocapture asr`。
//
// 提示词控制输出风格（context 通道）：chat 模板 system 段可携带用户提示词
// （`context`）。基座模型只把它当识别先验/热词偏置（Qwen3-ASR 被刻意训练成
// 不遵循 prompt 里的自然语言指令）；微调模型包（Release 一键下载，目录带
// `shuo-model.json` 标记）则把它当任务指令执行——同一个模型一趟完成识别 +
// 按要求的风格输出文本，无需 LLM 后处理。训练与推理的模板逐 token 一致（见
// 模型包内 `shuo-model.json` 的 prompts 记录与 `prompt_ids` 单测）。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::time::Instant;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::mel::{self, HOP, N_MELS, SAMPLE_RATE};

pub const VOCAB_SIZE: usize = 151_936;
pub const HIDDEN: usize = 1024;
pub const ENC_HIDDEN: usize = 896;
pub const NUM_LAYERS: usize = 28;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const CHUNK_SIZE: usize = 100;

pub const AUDIO_START_ID: i64 = 151_669;
pub const AUDIO_END_ID: i64 = 151_670;
pub const AUDIO_PAD_ID: i64 = 151_676;
pub const IM_START_ID: i64 = 151_644;
pub const IM_END_ID: i64 = 151_645;
pub const ENDOFTEXT_ID: i64 = 151_643;
pub const NEWLINE_ID: i64 = 198;

// ─────────────────────── 输出风格：提示词预设 ───────────────────────
//
// 这些短提示词进 chat 模板 system 段（context 通道），是「提示词控制输出
// 风格」的任务开关。字符串与模型训练时的提示词 **逐字一致**（模型包
// `shuo-model.json` 的 `prompts` 字段记录了每把钥匙；改动前先用它对表）。
// 微调数据的 prompt 字段用的就是这批字符串；推理端换了措辞=模型没学过的任务。

pub const PRESET_VERBATIM: &str = "verbatim";
pub const PRESET_POLISH_ZH: &str = "polish_zh";
pub const PRESET_POLISH_EN: &str = "polish_en";
pub const PRESET_TRANSLATE_ZH: &str = "translate_zh";
pub const PRESET_TRANSLATE_EN: &str = "translate_en";
pub const PRESET_BRIEF_ZH: &str = "brief_zh";
pub const PRESET_DETAIL_ZH: &str = "detail_zh";
pub const PRESET_CUSTOM: &str = "custom";

/// 预设 key → system 段提示词全文（`verbatim` = 空串 = 纯转写，基座原生行为）。
pub fn style_preset_prompt(key: &str) -> Option<&'static str> {
  Some(match key {
    PRESET_VERBATIM => "",
    PRESET_POLISH_ZH => "转为书面文本：修正标点，去除口癖，保持原意。",
    PRESET_POLISH_EN => "Rewrite as clean written text: fix punctuation, remove fillers, keep the meaning.",
    PRESET_TRANSLATE_ZH => "translate to Chinese",
    PRESET_TRANSLATE_EN => "translate to English",
    PRESET_BRIEF_ZH => "简略风格：压缩为要点，尽量简短。",
    PRESET_DETAIL_ZH => "详细风格：保留全部细节，输出完整书面文本。",
    _ => return None,
  })
}

/// (key, 中文名, 英文名)——前端下拉框展示用。
pub fn style_preset_list() -> Vec<(&'static str, &'static str, &'static str)> {
  vec![
    (PRESET_VERBATIM, "逐字转写（不优化）", "Verbatim (no polish)"),
    (PRESET_POLISH_ZH, "书面化 - 标点/去口癖", "Written text — punctuation/fillers"),
    (PRESET_POLISH_EN, "Written polish (EN)", "Written polish (EN)"),
    (PRESET_TRANSLATE_ZH, "中文（翻译）", "Chinese (translate)"),
    (PRESET_TRANSLATE_EN, "英文（翻译）", "English (translate)"),
    (PRESET_BRIEF_ZH, "简略 - 压缩要点", "Brief — condense key points"),
    (PRESET_DETAIL_ZH, "详细 - 保留全部细节", "Detailed — keep all details"),
  ]
}

/// 默认自定义提示词（`style_preset == "custom"` 时的初始值）。
pub fn default_style_prompt() -> String {
  "转为书面文本：修正标点，去除口癖，保持原意。".to_string()
}

#[derive(Debug, Clone)]
pub struct AsrResult {
  pub text: String,
  pub language: String,
  pub audio_s: f64,
  pub asr_s: f64,
  pub tokens: usize,
  pub chunks: usize,
}

pub struct AsrPipeline {
  conv: Session,
  transformer: Session,
  dec_init: Session,
  dec_step: Session,
  embed: Vec<f32>,
  tokenizer: Tokenizer,
}

fn open_session(path: &Path, threads: usize) -> Result<Session, String> {
  Session::builder()
    .map_err(|e| format!("创建 onnxruntime 会话失败：{e}"))?
    .with_intra_threads(threads.max(1))
    .map_err(|e| format!("设置推理线程数失败：{e}"))?
    .with_inter_threads(1)
    .map_err(|e| format!("设置并行线程数失败：{e}"))?
    .with_intra_op_spinning(false)
    .map_err(|e| format!("关闭自旋等待失败：{e}"))?
    .commit_from_file(path)
    .map_err(|e| format!("加载模型 {} 失败：{e}", path.display()))
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
  let bytes = std::fs::read(path).map_err(|e| format!("读取 {} 失败：{e}", path.display()))?;
  if bytes.len() % 4 != 0 {
    return Err(format!("{} 长度不是 4 的倍数", path.display()));
  }
  Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

impl AsrPipeline {
  /// `model_root` 是 Qwen3-ASR-0.6B-ONNX-CPU 目录（内含 `onnx_models/` 与 `tokenizer.json`）。
  pub fn load(model_root: &Path, threads: usize) -> Result<Self, String> {
    let onnx = model_root.join("onnx_models");
    let pick = |name: &str| -> PathBuf {
      let p = onnx.join(name);
      p
    };
    let dec_init_path = if pick("decoder_init.int8.onnx").exists() {
      pick("decoder_init.int8.onnx")
    } else {
      pick("decoder_init.onnx")
    };
    let dec_step_path = if pick("decoder_step.int8.onnx").exists() {
      pick("decoder_step.int8.onnx")
    } else {
      pick("decoder_step.onnx")
    };

    let conv = open_session(&pick("encoder_conv.onnx"), threads)?;
    let transformer = open_session(&pick("encoder_transformer.onnx"), threads)?;
    let dec_init = open_session(&dec_init_path, threads)?;
    let dec_step = open_session(&dec_step_path, threads)?;

    let embed = read_f32(&onnx.join("embed_tokens.bin"))?;
    if embed.len() != VOCAB_SIZE * HIDDEN {
      return Err(format!(
        "embed_tokens.bin 大小不符：{} 个 float，期望 {}",
        embed.len(),
        VOCAB_SIZE * HIDDEN
      ));
    }

    let tok_path = if onnx.join("tokenizer.json").exists() {
      onnx.join("tokenizer.json")
    } else {
      model_root.join("tokenizer.json")
    };
    let tokenizer =
      Tokenizer::from_file(&tok_path).map_err(|e| format!("加载 {} 失败：{e}", tok_path.display()))?;

    Ok(AsrPipeline { conv, transformer, dec_init, dec_step, embed, tokenizer })
  }

  fn encode_ids(&self, text: &str) -> Result<Vec<i64>, String> {
    let enc = self.tokenizer.encode(text, false).map_err(|e| format!("分词失败：{e}"))?;
    Ok(enc.get_ids().iter().map(|&v| v as i64).collect())
  }

  fn decode_ids(&self, ids: &[i64]) -> Result<String, String> {
    let u32_ids: Vec<u32> = ids.iter().map(|&v| v.max(0) as u32).collect();
    self
      .tokenizer
      .decode(&u32_ids, true)
      .map_err(|e| format!("解码失败：{e}"))
  }

  fn build_prompt(
    &self,
    num_audio_tokens: usize,
    language: Option<&str>,
    context: &str,
  ) -> Result<Vec<i64>, String> {
    prompt_ids(
      &|t| self.encode_ids(t),
      num_audio_tokens,
      language,
      context,
    )
  }

  /// encoder：mel [N_MELS, frames] → 音频特征 [n_tokens, HIDDEN]
  fn encode_audio(&mut self, mel_data: &[f32], frames: usize) -> Result<Vec<f32>, String> {
    let chunks = frames.div_ceil(CHUNK_SIZE);
    let mut chunk_lens = Vec::with_capacity(chunks);
    for i in 0..chunks {
      let start = i * CHUNK_SIZE;
      chunk_lens.push((frames - start).min(CHUNK_SIZE));
    }
    let max_len = *chunk_lens.iter().max().unwrap_or(&1);

    let mut padded = vec![0.0f32; chunks * N_MELS * max_len];
    for (c, &cl) in chunk_lens.iter().enumerate() {
      let start = c * CHUNK_SIZE;
      for m in 0..N_MELS {
        let src = &mel_data[m * frames + start..m * frames + start + cl];
        let dst = &mut padded[(c * N_MELS + m) * max_len..(c * N_MELS + m) * max_len + cl];
        dst.copy_from_slice(src);
      }
    }

    let conv_len = {
      let mut l = max_len;
      for _ in 0..3 {
        l = (l - 1) / 2 + 1;
      }
      l
    };

    let mel_t = Tensor::from_array(([chunks as i64, 1, N_MELS as i64, max_len as i64], padded))
      .map_err(|e| format!("构造 padded_mel_chunks 失败：{e}"))?;
    let out = self
      .conv
      .run(ort::inputs!["padded_mel_chunks" => mel_t])
      .map_err(|e| format!("encoder_conv 推理失败：{e}"))?;
    let (shape, data) = out["chunk_features"]
      .try_extract_tensor::<f32>()
      .map_err(|e| format!("读取 chunk_features 失败：{e}"))?;
    let dims: Vec<i64> = shape.iter().copied().collect();
    if dims.len() != 3 || dims[2] as usize != ENC_HIDDEN {
      return Err(format!("chunk_features 形状异常：{dims:?}"));
    }
    let conv_len = conv_len.min(dims[1] as usize);

    // 按每段的有效长度拼接（去掉 padding）
    let lens: Vec<usize> = chunk_lens
      .iter()
      .map(|&l| {
        let mut v = l;
        for _ in 0..3 {
          v = (v - 1) / 2 + 1;
        }
        v.min(conv_len)
      })
      .collect();
    let total: usize = lens.iter().sum();
    let mut features = Vec::with_capacity(total * ENC_HIDDEN);
    for (c, &l) in lens.iter().enumerate() {
      let base = c * conv_len * ENC_HIDDEN;
      features.extend_from_slice(&data[base..base + l * ENC_HIDDEN]);
    }

    let hs = Tensor::from_array(([total as i64, ENC_HIDDEN as i64], features))
      .map_err(|e| format!("构造 hidden_states 失败：{e}"))?;
    let mask = Tensor::from_array((
      [1i64, 1, total as i64, total as i64],
      vec![0.0f32; total * total],
    ))
    .map_err(|e| format!("构造 attention_mask 失败：{e}"))?;
    let out = self
      .transformer
      .run(ort::inputs!["hidden_states" => hs, "attention_mask" => mask])
      .map_err(|e| format!("encoder_transformer 推理失败：{e}"))?;
    let (shape, feat) = out["encoder_output"]
      .try_extract_tensor::<f32>()
      .map_err(|e| format!("读取 encoder_output 失败：{e}"))?;
    let dims: Vec<i64> = shape.iter().copied().collect();
    let n = dims.first().copied().unwrap_or(0) as usize;
    if n != total || feat.len() < total * HIDDEN {
      return Err(format!("encoder_output 形状异常：{dims:?}"));
    }
    Ok(feat[..total * HIDDEN].to_vec())
  }

  fn fuse_embeds(&self, ids: &[i64], audio: &[f32]) -> Result<(Vec<f32>, usize), String> {
    let seq = ids.len();
    let mut embeds = Vec::with_capacity(seq * HIDDEN);
    let mut audio_idx = 0usize;
    for &id in ids {
      if id == AUDIO_PAD_ID {
        let start = audio_idx * HIDDEN;
        if start + HIDDEN > audio.len() {
          return Err(format!("音频 token 数与特征数不匹配（token {audio_idx}）"));
        }
        embeds.extend_from_slice(&audio[start..start + HIDDEN]);
        audio_idx += 1;
      } else {
        let id = id as usize;
        if id >= VOCAB_SIZE {
          return Err(format!("token id 越界：{id}"));
        }
        let start = id * HIDDEN;
        embeds.extend_from_slice(&self.embed[start..start + HIDDEN]);
      }
    }
    Ok((embeds, seq))
  }

  fn transcribe_chunk(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    max_new_tokens: usize,
  ) -> Result<(String, String, usize), String> {
    let (mel_data, frames) = mel::log_mel(wav);
    if frames == 0 {
      return Err("音频太短".into());
    }
    let audio = self.encode_audio(&mel_data, frames)?;
    let n_audio = audio.len() / HIDDEN;
    let ids = self.build_prompt(n_audio, language, context)?;
    let (embeds, seq) = self.fuse_embeds(&ids, &audio)?;
    drop(audio);

    let emb_t = Tensor::from_array(([1i64, seq as i64, HIDDEN as i64], embeds))
      .map_err(|e| format!("构造 input_embeds 失败：{e}"))?;
    let pos: Vec<i64> = (0..seq as i64).collect();
    let pos_t = Tensor::from_array(([1i64, seq as i64], pos))
      .map_err(|e| format!("构造 position_ids 失败：{e}"))?;

    let mut out = self
      .dec_init
      .run(ort::inputs!["input_embeds" => emb_t, "position_ids" => pos_t])
      .map_err(|e| format!("decoder_init 推理失败：{e}"))?;
    let (lshape, logits) = out["logits"]
      .try_extract_tensor::<f32>()
      .map_err(|e| format!("读取 logits 失败：{e}"))?;
    let ldims: Vec<i64> = lshape.iter().copied().collect();
    let vocab = *ldims.last().unwrap_or(&(VOCAB_SIZE as i64)) as usize;
    let mut next = argmax(&logits[logits.len() - vocab..]) as i64;
    let mut generated = vec![next];

    let mut keys = out
      .remove("present_keys")
      .ok_or_else(|| "decoder_init 缺少 present_keys".to_string())?;
    let mut values = out
      .remove("present_values")
      .ok_or_else(|| "decoder_init 缺少 present_values".to_string())?;
    drop(out);
    let mut past_len = seq;

    while generated.len() < max_new_tokens {
      if next == IM_END_ID || next == ENDOFTEXT_ID {
        break;
      }
      let start = (next as usize) * HIDDEN;
      let token_embed = self.embed[start..start + HIDDEN].to_vec();
      let emb = Tensor::from_array(([1i64, 1, HIDDEN as i64], token_embed))
        .map_err(|e| format!("构造 token embeds 失败：{e}"))?;
      let pos = Tensor::from_array(([1i64, 1], vec![past_len as i64]))
        .map_err(|e| format!("构造 step position_ids 失败：{e}"))?;

      let inputs: Vec<(&str, SessionInputValue)> = vec![
        ("input_embeds", emb.into()),
        ("position_ids", pos.into()),
        ("past_keys", SessionInputValue::from(keys)),
        ("past_values", SessionInputValue::from(values)),
      ];
      let mut out = self
        .dec_step
        .run(inputs)
        .map_err(|e| format!("decoder_step 推理失败：{e}"))?;
      let (lshape, logits) = out["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取 step logits 失败：{e}"))?;
      let ldims: Vec<i64> = lshape.iter().copied().collect();
      let vocab = *ldims.last().unwrap_or(&(VOCAB_SIZE as i64)) as usize;
      next = argmax(&logits[logits.len() - vocab..]) as i64;
      generated.push(next);
      keys = out
        .remove("present_keys")
        .ok_or_else(|| "decoder_step 缺少 present_keys".to_string())?;
      values = out
        .remove("present_values")
        .ok_or_else(|| "decoder_step 缺少 present_values".to_string())?;
      past_len += 1;
    }

    if matches!(generated.last(), Some(&IM_END_ID) | Some(&ENDOFTEXT_ID)) {
      generated.pop();
    }
    let raw = self.decode_ids(&generated)?;
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
    Ok((text, lang, generated.len()))
  }

  /// 转写一段 16 kHz 单声道音频；长音频按静音切分。
  /// `context` 进 chat 模板 system 段（输出风格任务开关；空串 = 原生转写）。
  pub fn transcribe(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    max_new_tokens: usize,
    chunk_sec: usize,
  ) -> Result<AsrResult, String> {
    let t0 = Instant::now();
    let audio_s = wav.len() as f64 / SAMPLE_RATE as f64;
    let splits = mel::silence_splits(wav, chunk_sec);
    if splits.is_empty() {
      let (text, lang, tokens) = self.transcribe_chunk(wav, language, context, max_new_tokens)?;
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
      let (text, l, n_tokens) = self.transcribe_chunk(
        &wav[bounds[i]..bounds[i + 1]],
        language,
        context,
        max_new_tokens,
      )?;
      let text = text.trim();
      if !text.is_empty() {
        texts.push(text.to_string());
      }
      if lang.is_empty() {
        lang = l;
      }
      tokens += n_tokens;
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

  /// 无分段（与 Python `_transcribe_chunk` 等价），降噪路径使用。
  pub fn transcribe_single(
    &mut self,
    wav: &[f32],
    language: Option<&str>,
    context: &str,
    max_new_tokens: usize,
  ) -> Result<AsrResult, String> {
    let t0 = Instant::now();
    let (text, lang, tokens) = self.transcribe_chunk(wav, language, context, max_new_tokens)?;
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

/// chat 模板 token 序列构造（纯函数，`encode` 为「文本片段 → token id」闭包）。
///
/// 模板与 HF `Qwen/Qwen3-ASR-0.6B` 的 `apply_chat_template`（训练侧
/// 训练侧 tokenizer 前缀）以及 llama.cpp 侧实测逐 token 一致：
///
/// ```text
/// <|im_start|>system\n{context}<|im_end|>\n
/// <|im_start|>user\n<|audio_bos|><|AUDIO|>×N<|audio_eos|><|im_end|>\n
/// <|im_start|>assistant\n[language X<asr_text>]
/// ```
///
/// context 为空时保持历史路径（`encode("system")` + 换行 token），数值与
/// Python 版逐 token 对齐（AGENTS.md 记录的既有保证不回归）；非空时
/// `system\n{context}` 作为单一片段整体编码——与训练端「整串渲染后一次
/// tokenize」在特殊 token 边界内的行为一致。
pub fn prompt_ids(
  encode: &dyn Fn(&str) -> Result<Vec<i64>, String>,
  num_audio_tokens: usize,
  language: Option<&str>,
  context: &str,
) -> Result<Vec<i64>, String> {
  let ctx = context.trim();
  let mut ids = vec![IM_START_ID];
  if ctx.is_empty() {
    ids.extend(encode("system")?);
    ids.push(NEWLINE_ID);
  } else {
    ids.extend(encode(&format!("system\n{ctx}"))?);
  }
  ids.extend([IM_END_ID, NEWLINE_ID]);
  ids.push(IM_START_ID);
  ids.extend(encode("user")?);
  ids.push(NEWLINE_ID);
  ids.push(AUDIO_START_ID);
  ids.extend(std::iter::repeat_n(AUDIO_PAD_ID, num_audio_tokens));
  ids.push(AUDIO_END_ID);
  ids.extend([IM_END_ID, NEWLINE_ID]);
  ids.push(IM_START_ID);
  ids.extend(encode("assistant")?);
  ids.push(NEWLINE_ID);
  if let Some(lang) = language {
    ids.extend(encode(&format!("language {lang}<asr_text>"))?);
  }
  Ok(ids)
}

fn argmax(v: &[f32]) -> usize {
  let mut best = 0usize;
  let mut best_v = f32::NEG_INFINITY;
  for (i, &x) in v.iter().enumerate() {
    if x > best_v {
      best_v = x;
      best = i;
    }
  }
  best
}

/// 帧数与 hop 的换算（调试用）
pub fn frames_for(samples: usize) -> usize {
  1 + samples / HOP
}

#[cfg(test)]
mod tests {
  use super::*;

  fn model_root() -> PathBuf {
    std::env::var("SHUO_ASR_DIR")
      .map(PathBuf::from)
      .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../Qwen3-ASR-0.6B-ONNX-CPU"))
  }

  /// 假编码器：把每个字符映射成固定 id（'a'→1），只为验证模板结构。
  fn fake_encode(text: &str) -> Result<Vec<i64>, String> {
    Ok(text.chars().map(|c| c as i64).collect())
  }

  #[test]
  fn prompt_ids_empty_context_matches_legacy_layout() {
    // 空 context 必须保持历史结构：im_start + "system" + \n + im_end + \n …
    let ids = prompt_ids(&fake_encode, 2, None, "").expect("ids");
    let mut expect = vec![IM_START_ID];
    expect.extend("system".chars().map(|c| c as i64));
    expect.extend([NEWLINE_ID, IM_END_ID, NEWLINE_ID, IM_START_ID]);
    expect.extend("user".chars().map(|c| c as i64));
    expect.push(NEWLINE_ID);
    expect.push(AUDIO_START_ID);
    expect.extend([AUDIO_PAD_ID, AUDIO_PAD_ID]);
    expect.push(AUDIO_END_ID);
    expect.extend([IM_END_ID, NEWLINE_ID, IM_START_ID]);
    expect.extend("assistant".chars().map(|c| c as i64));
    expect.push(NEWLINE_ID);
    assert_eq!(ids, expect);
  }

  #[test]
  fn prompt_ids_context_goes_into_system_segment() {
    // context 非空：`system\n{ctx}` 整体作为单一片段编码（与训练端一致），
    // 且只出现在 system 段，user/assistant 段结构不变。
    let ids = prompt_ids(&fake_encode, 1, None, "translate to English").expect("ids");
    let mut expect = vec![IM_START_ID];
    expect.extend("system\ntranslate to English".chars().map(|c| c as i64));
    expect.extend([IM_END_ID, NEWLINE_ID, IM_START_ID]);
    expect.extend("user".chars().map(|c| c as i64));
    expect.push(NEWLINE_ID);
    expect.push(AUDIO_START_ID);
    expect.push(AUDIO_PAD_ID);
    expect.push(AUDIO_END_ID);
    expect.extend([IM_END_ID, NEWLINE_ID, IM_START_ID]);
    expect.extend("assistant".chars().map(|c| c as i64));
    expect.push(NEWLINE_ID);
    assert_eq!(ids, expect);
    // 首尾空白要修剪（用户自定义提示词常带换行）
    let trimmed = prompt_ids(&fake_encode, 1, None, "\n  translate to English \n").expect("ids");
    assert_eq!(trimmed, ids);
  }

  #[test]
  fn prompt_ids_forced_language_suffix() {
    let ids = prompt_ids(&fake_encode, 1, Some("Chinese"), "转为书面文本").expect("ids");
    let tail: Vec<i64> = "language Chinese<asr_text>".chars().map(|c| c as i64).collect();
    assert!(ids.ends_with(&tail));
    // context 仍然进 system 段
    let head: Vec<i64> = "system\n转为书面文本".chars().map(|c| c as i64).collect();
    assert_eq!(ids[1..1 + head.len()], head);
  }

  #[test]
  fn style_presets_are_stable() {
    // 预设 key/提示词是训练数据的一部分：与模型包 shuo-model.json 的 prompts 记录
    // 逐字一致，改动前先对表（详见文件头注释）。
    assert_eq!(style_preset_prompt(PRESET_VERBATIM), Some(""));
    assert_eq!(
      style_preset_prompt(PRESET_TRANSLATE_ZH),
      Some("translate to Chinese")
    );
    assert_eq!(
      style_preset_prompt(PRESET_TRANSLATE_EN),
      Some("translate to English")
    );
    // 风格任务（简略/详细）：与训练侧的长度比判据配套
    assert_eq!(
      style_preset_prompt(PRESET_BRIEF_ZH),
      Some("简略风格：压缩为要点，尽量简短。")
    );
    assert_eq!(
      style_preset_prompt(PRESET_DETAIL_ZH),
      Some("详细风格：保留全部细节，输出完整书面文本。")
    );
    assert!(style_preset_prompt("nonexistent").is_none());
    let keys: Vec<&str> = style_preset_list().iter().map(|(k, _, _)| *k).collect();
    assert_eq!(
      keys,
      vec![
        PRESET_VERBATIM,
        PRESET_POLISH_ZH,
        PRESET_POLISH_EN,
        PRESET_TRANSLATE_ZH,
        PRESET_TRANSLATE_EN,
        PRESET_BRIEF_ZH,
        PRESET_DETAIL_ZH,
      ]
    );
    // custom 不在列表里（前端单独追加），但解析必须走 style_prompt
    assert!(style_preset_prompt(PRESET_CUSTOM).is_none());
  }

  #[test]
  fn transcribe_reference_wav() {
    let root = model_root();
    if !root.join("onnx_models/encoder_conv.onnx").exists() {
      eprintln!("跳过：{} 不存在", root.display());
      return;
    }
    let wav_path = root.join("test_audio/librispeech_0_1089_0.wav");
    let mut reader = hound_lite::Wav::open(&wav_path).expect("wav");
    let wav16 = reader.read_i16_mono();
    let wav: Vec<f32> = wav16.iter().map(|&s| s as f32 / 32768.0).collect();
    // 测试音频本身就是 16 kHz
    let mut p = AsrPipeline::load(&root, 6).expect("load");
    let r = p.transcribe(&wav, None, "", 512, 30).expect("transcribe");
    eprintln!("TEXT={}", r.text);
    eprintln!("LANG={} tokens={} asr_s={:.2}", r.language, r.tokens, r.asr_s);
    assert!(!r.text.trim().is_empty());
  }

  /// 真实模型冒烟（仅 SHUO_ASR_DIR 存在时跑，CI 的 ort-e2e 模式使用）：
  /// 基座模型带 context 也必须正常出文本——context 只是 system 段偏置，
  /// 不能把原生转写打挂（微调包的「按提示词输出风格」行为另在
  /// 模型包验证流程实测）。
  #[test]
  fn transcribe_with_context_smoke() {
    let root = model_root();
    if !root.join("onnx_models/encoder_conv.onnx").exists() {
      eprintln!("跳过：{} 不存在", root.display());
      return;
    }
    let wav_path = root.join("test_audio/librispeech_1_1089_1.wav");
    let mut reader = hound_lite::Wav::open(&wav_path).expect("wav");
    let wav16 = reader.read_i16_mono();
    let wav: Vec<f32> = wav16.iter().map(|&s| s as f32 / 32768.0).collect();
    let mut p = AsrPipeline::load(&root, 6).expect("load");
    let plain = p.transcribe_single(&wav, None, "", 512).expect("plain");
    let ctx = p
      .transcribe_single(&wav, None, "translate to English", 512)
      .expect("ctx");
    eprintln!("PLAIN={}", plain.text);
    eprintln!("CTX  ={}", ctx.text);
    assert!(!plain.text.trim().is_empty());
    assert!(!ctx.text.trim().is_empty());
  }
}

/// 极简 WAV 读取（仅测试/调试用，避免引入额外依赖）。
pub mod hound_lite {
  use std::fs::File;
  use std::io::{Read, Seek, SeekFrom};
  use std::path::Path;

  pub struct Wav {
    pub channels: u16,
    pub rate: u32,
    data: Vec<i16>,
  }

  impl Wav {
    pub fn open(path: &Path) -> Result<Self, String> {
      let mut f = File::open(path).map_err(|e| e.to_string())?;
      let mut hdr = [0u8; 12];
      f.read_exact(&mut hdr).map_err(|e| e.to_string())?;
      if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err("不是 RIFF/WAVE 文件".into());
      }
      let (mut channels, mut rate, mut bits) = (1u16, 16000u32, 16u16);
      loop {
        let mut ch = [0u8; 8];
        if f.read_exact(&mut ch).is_err() {
          break;
        }
        let id = &ch[0..4];
        let size = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]) as usize;
        match id {
          b"fmt " => {
            let mut fmt = vec![0u8; size];
            f.read_exact(&mut fmt).map_err(|e| e.to_string())?;
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits = u16::from_le_bytes([fmt[14], fmt[15]]);
          }
          b"data" => {
            let mut data = vec![0u8; size];
            f.read_exact(&mut data).map_err(|e| e.to_string())?;
            if bits != 16 {
              return Err(format!("仅支持 16-bit WAV（当前 {bits}）"));
            }
            let samples = data
              .chunks_exact(2)
              .map(|b| i16::from_le_bytes([b[0], b[1]]))
              .collect();
            return Ok(Wav { channels, rate, data: samples });
          }
          _ => {
            f.seek(SeekFrom::Current(size as i64)).map_err(|e| e.to_string())?;
          }
        }
      }
      Err("没有 data 块".into())
    }

    pub fn read_i16_mono(&mut self) -> Vec<i16> {
      if self.channels <= 1 {
        return self.data.clone();
      }
      self
        .data
        .chunks(self.channels as usize)
        .map(|c| (c.iter().map(|&v| v as i32).sum::<i32>() / self.channels as i32) as i16)
        .collect()
    }

    pub fn rate(&self) -> u32 {
      self.rate
    }
  }
}

#[cfg(test)]
mod compare_tests {
  use super::*;

  #[test]
  fn dump_mel_and_encoder() {
    let root = std::env::var("SHUO_ASR_DIR")
      .map(std::path::PathBuf::from)
      .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../Qwen3-ASR-0.6B-ONNX-CPU"));
    if !root.join("onnx_models/encoder_conv.onnx").exists() {
      return;
    }
    let mut r = hound_lite::Wav::open(&root.join("test_audio/librispeech_0_1089_0.wav")).unwrap();
    let wav: Vec<f32> = r.read_i16_mono().iter().map(|&s| s as f32 / 32768.0).collect();
    let (mel_data, frames) = mel::log_mel(&wav);
    // 调试转储目录 CI 上不存在，先建再写
    std::fs::create_dir_all("/tmp/opencode").unwrap();
    std::fs::write("/tmp/opencode/mel_rs.f32", mel_data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    eprintln!("frames={frames}");
    let mut p = AsrPipeline::load(&root, 6).unwrap();
    let audio = p.encode_audio(&mel_data, frames).unwrap();
    std::fs::write("/tmp/opencode/enc_rs.f32", audio.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    eprintln!("enc len={} tokens={}", audio.len(), audio.len()/HIDDEN);
  }
}
