# 说 · Shuo

本地语音识别 GUI，基于 Qwen3-ASR-0.6B ONNX CPU 推理 + Qwen3.5-2B ONNX 文本优化。无需 GPU，无需 PyTorch，无需云端。

On-device speech recognition GUI powered by Qwen3-ASR-0.6B ONNX CPU pipeline with Qwen3.5-2B ONNX text post-processing. No GPU, no PyTorch, no cloud.

## 快速开始 / Quick Start

| 平台 / Platform | 指南 / Guide | ASR 模型下载 |
|---|---|---|
| Windows | [中文](./docs/windows.md) / [English](./docs/windows-en.md) | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) |
| Linux | [中文](./docs/linux.md) / [English](./docs/linux-en.md) | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) |

下载后解压到项目根目录。

## 项目文件 / Project Structure

```
├── docs/                       # 文档 Documentation
├── locales/                    # 翻译文件 Translation files
├── Qwen3-ASR-0.6B-ONNX-CPU/   # ASR 模型 ASR model
├── Qwen3.5-2B-ONNX-OPT/       # LLM 文本优化模型 LLM text model
├── onnx_infer/                # ONNX 模型与推理模块
│   ├── asr.py                 # ASR 推理管线 ASR inference pipeline (0.6B)
│   ├── denoise.py             # 音频降噪 Audio denoiser
│   ├── denoise.onnx           # 降噪模型 Denoiser model
│   └── llm.py                 # LLM 文本优化 LLM text post-processing
├── i18n/                      # 国际化模块 i18n module
│   ├── __init__.py            # 运行时 i18n Runtime i18n
│   └── build_i18n_local.py    # 生成/提取语言文件 Generate/extract locale files
├── asr_gui.py                 # 语音识别 GUI Speech recognition GUI
├── hotkey.py                  # 全局快捷键 Global hotkey
├── locales/                   # 翻译文件 Translation files (build generated)
```

## 配置 / Configuration

应用启动后自动创建 `~/.shuo/` 目录：

App creates `~/.shuo/` on first launch:

| 文件 File | 说明 Description |
|---|---|
| `config.json` | 语言、快捷键、自动输入 Language, hotkey, auto-type |
| `history.json` | 识别历史（最多 500 条）History (max 500) |
| `shuo.log` | 运行日志（UTF-8）Runtime log (UTF-8) |

- 默认快捷键：鼠标侧键（后退），可在左上角「快捷键」按钮自定义
- Default hotkey: mouse side button (back), customizable via top-left button
- 右上角下拉框切换中文 / English
- Language switch: top-right dropdown

## 依赖 / Dependencies

| 库 Library | 用途 Purpose |
|---|---|
| PySide6 | GUI 框架 Framework |
| qtawesome | Font Awesome 图标 Icons |
| pynput | 全局热键监听 Global hotkey listener |
| PyAudio | 音频录制 Audio recording |
| onnxruntime | ONNX 模型推理 Model inference |
| numpy | 数值计算 Numerical computing |
| tokenizers | 文本分词 Text tokenization |

## 许可证 / License

代码 Code：GPL-3.0

### Qt / PySide6

本应用使用 PySide6（Qt for Python），依据 **LGPL v3** 协议发布。可在专有软件中动态链接，无需开源。

This app uses PySide6 (Qt for Python) under **LGPL v3**. Dynamic linking in proprietary software is permitted without open-sourcing your code.

完整协议 Full license: https://www.gnu.org/licenses/lgpl-3.0.html

### 第三方库 / Third-party Libraries

| 库 Library | 协议 License |
|---|---|
| qtawesome | MIT |
| pynput | LGPL-3.0 |
| PyAudio | MIT |
| onnxruntime | MIT |
| numpy | BSD-3 |
| tokenizers | Apache-2.0 |

### 模型 / Models

模型权重版权归原作者所有，详见 [NOTICE](./NOTICE)。

Model weights: copyright belongs to original authors. See [NOTICE](./NOTICE).

文本优化模型 Text model：[Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT)（Apache 2.0）
