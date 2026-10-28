// 说 · Shuo — ASR 公共契约（识别结果 / 输出风格预设 / 测试辅助）
//
// 推理实现只有一条路径：`gguf_asr.rs`（llama.cpp/GGUF，1.7B，CPU）。
// ONNX ASR 后端（Daumee 0.6B 包）已于 2026-10-28 移除；`ort` 现在只服务
// 降噪模型（`denoise.rs` + 内置 `resources/denoise.onnx`）。
//
// 提示词控制输出风格（context 通道）：chat 模板 system 段可携带用户提示词
// （`context`）。基座模型只把它当识别先验/热词偏置（Qwen3-ASR 被刻意训练成
// 不遵循 prompt 里的自然语言指令）；微调模型包（ModelScope 一键下载，目录带
// `shuo-model.json` 标记）则把它当任务指令执行——同一个模型一趟完成识别 +
// 按要求的风格输出文本，无需 LLM 后处理。推理端的模板在 `gguf_asr.rs`
// （llama.cpp 文本模板）；本文件底部的 `prompt_contract` 测试保留 Qwen3-ASR
// chat 模板的 token 级布局作为对照（训练/推理措辞必须逐字一致）。
//
// SPDX-License-Identifier: GPL-3.0-or-later

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

/// 一次识别的结果（GGUF/llama.cpp 路径产出，worker 直接消费）。
#[derive(Debug, Clone)]
pub struct AsrResult {
  pub text: String,
  pub language: String,
  pub audio_s: f64,
  pub asr_s: f64,
  pub tokens: usize,
  pub chunks: usize,
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
mod tests {
  use super::*;

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
}

/// Qwen3-ASR chat 模板的 token 级布局（契约测试）。
///
/// ONNX 后端移除后不再有代码调用它，但 `gguf_asr.rs` 的文本模板与训练端的
/// `chat_template.jinja` 必须与此布局等价（system 段放 context、音频占位、
/// assistant 段 `language X<asr_text>`）；留作回归对照，避免改措辞/顺序。
#[cfg(test)]
mod prompt_contract {
  pub const AUDIO_START_ID: i64 = 151_669;
  pub const AUDIO_END_ID: i64 = 151_670;
  pub const AUDIO_PAD_ID: i64 = 151_676;
  pub const IM_START_ID: i64 = 151_644;
  pub const IM_END_ID: i64 = 151_645;
  pub const NEWLINE_ID: i64 = 198;

  /// 假编码器：把每个字符映射成固定 id（'a'→1），只为验证模板结构。
  fn fake_encode(text: &str) -> Result<Vec<i64>, String> {
    Ok(text.chars().map(|c| c as i64).collect())
  }

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
}
