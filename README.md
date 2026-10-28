# 说 · Shuo

本地语音识别 GUI，基于微调 Qwen3-ASR（1.7B，llama.cpp/GGUF）：**提示词控制输出风格**的语音转写——识别与文本风格处理在同一次前向里完成。无需云端，无需 PyTorch，无需 Python。

On-device speech recognition GUI powered by a fine-tuned Qwen3-ASR (1.7B, llama.cpp/GGUF): transcription with **prompt-controlled output style** — recognition and text styling happen in a single pass. No cloud, no PyTorch, no Python.

## 安装 / 运行

从 Release 获取安装包（Windows NSIS / Linux deb + AppImage），或本地构建：

```bash
# 1)（可选）模型放到仓库根目录 / ~/.shuo/models/，或启动后一键下载

# 2) 运行
cd src-tauri
cargo run --release

# 3) 打包（Linux；llama.cpp 静态链接需要 clang/libclang）
cargo tauri build --bundles deb appimage

# 4) 打包（Windows，需在 Windows 上）
powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1
```

- **全局热键**：两个独立热键、各自带开关——
  - 主热键（默认 `F2`）→ 识别后用 **Ctrl+V** 上屏，适合图形界面；
  - 终端热键（默认 `F3`）→ 识别后用 **Ctrl+Shift+V** 上屏，适合终端 / TUI。
- **输出风格**（设置 → 输出风格）：微调模型包按 system 段提示词直接输出目标风格
  文本；预设清单即 UI 下拉框内容（`src-tauri/src/asr.rs` 的 `style_preset_list()`，
  提示词与模型包 `shuo-model.json` 的 `prompts` 记录逐字对表）。
- **热键实现（Linux）**：优先走 `xdg-desktop-portal` 的 **GlobalShortcuts**（GNOME/KDE Wayland 官方通道，首次需在系统对话框里允许一次）；portal 不可用或要绑鼠标侧键时用 **evdev**（`/dev/input/event*`，需要 input 组权限或 `70-shuo-hotkey.rules`，deb 已自带）。
- **自动输入（Linux）**：剪贴板 + 粘贴快捷键，快捷键由触发录音的热键决定（Ctrl+V / Ctrl+Shift+V）。**首选 uinput 虚拟键盘**（不需要任何授权框，deb 的 udev 规则会放开 `/dev/uinput`）；uinput 不可用时退回 RemoteDesktop 门户（持久化 restore token，被撤回会自动重新申请）。
- **状态胶囊（Linux）**：屏幕底部（任务栏上方）显示「录音中 + 波形 / 识别中 + 转圈」，XWayland 小窗置顶、不抢焦点、鼠标穿透；不需要 GNOME 扩展，也不需要注销重登录。
- **自动输入（Windows）**：`SendInput` + `KEYEVENTF_UNICODE`，不经过剪贴板，终端 / Vim 也能用。

## 输出风格 / Output Style

用户设置的提示词进 ASR chat 模板 system 段（context 通道），微调模型按提示词
**直接输出目标风格的文本**——识别与文本处理一步完成：

* 风格预设：逐字转写 / 中文书面化 / 英文书面化 / 中文（翻译）/ 英文（翻译）/ **简略（压缩要点）/ 详细（保留全部细节）** / 自定义提示词；
* 基座模型不执行 prompt 指令（官方防指令注入设计），**必须配合微调模型包**
  （目录带 `shuo-model.json` 标记）才能兑现风格开关；基座包上运行不报错，
  提示词退化为识别偏置（选「逐字转写」预设可完全消除偏置）；
* 微调模型包通过 GitHub Release 分发，应用内**一键下载**（见下）。

> 本仓库是「调用模型的应用程序」：微调产线（数据 / LoRA 训练 / 评测 / GGUF 打包）
> 不在本仓库内，模型包以 Release 资产形式发布。

## 获取模型 / Get the Model

首次启动缺模型时，横幅与「设置 → 模型与引擎」都有**下载模型**按钮：
从项目 GitHub Release 拉取微调模型包（约 1.4 GB）到
`~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/`，支持断点续传，完成后自动重载引擎。
下载走 HTTPS，证书校验使用**操作系统信任库**（企业/本地代理的 TLS 拦截证书
装进系统后即可正常下载）。

> **下载源**：当前指向 `OrientCOMPASS/shuo` 的 pre-release `model-tuned-1.7B-pre`
> （`src-tauri/src/model_download.rs` 顶部两个常量，上游可自行改到自己的 Release）。

也可手动放置模型（查找顺序：配置项 `asr_model_dir` → `SHUO_MODEL_DIR` →
`~/.shuo/models/` → 可执行文件上溯 → 仓库目录）：

| 后端 | 模型目录 | 说明 |
|---|---|---|
| `gguf`（优先） | `Qwen3-ASR-1.7B-GGUF-tuned/`（微调模型包，一键下载即是它）或 `Qwen3-ASR-1.7B-GGUF/`（官方 ggml-org 包，≈2.4 GB） | 目录内 `mmproj` 前缀 `.gguf` = 音频编码器，其余最大 `.gguf` = LM |
| `onnx`（后备） | `Qwen3-ASR-0.6B-ONNX-CPU[-tuned]/` | Daumee 包（`hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU`）；ort 另承担降噪模型 |

配置 `asr_backend`：`auto`（默认，有 GGUF 模型优先 GGUF）/ `onnx` / `gguf`；
所选后端模型缺失时自动回退另一个可用后端（日志说明）。UI 状态 chip 显示当前后端。

## 项目文件 / Project Structure

```
├── src-tauri/                       # Tauri 3 应用（Rust 推理 + 静态前端）
│   ├── src/                         #   ASR（GGUF 主 / ONNX 后备）/ 模型下载 / 降噪 / 音频 / 热键 / 上屏
│   ├── ui/                          #   index.html / style.css / app.js / locales（27 语言）
│   ├── resources/denoise.onnx       #   降噪模型（打包进 deb/NSIS）
│   ├── linux/                       #   udev 规则（evdev 热键后备）
│   ├── build.ps1                    #   Windows NSIS 打包
│   └── tauri.conf.json              #   deb / AppImage / NSIS 配置
├── docs/                            # 文档（ONNX 模型卡、scaffold 对照等）
├── linux/                           # 状态胶囊辅助脚本（shuo-overlay.py）
├── scripts/                         # 发布脚本（release.sh / set-version.sh）
└── .github/workflows/               # CI（rust 双平台 + UI 断言）与 Release 构建
```

## 配置 / Configuration

应用启动后自动创建 `~/.shuo/` 目录：

| 文件 File | 说明 Description |
|---|---|
| `config.json` | 语言、快捷键、模型目录、后端选择、风格提示词（`style_preset`/`style_prompt`）Language, hotkey, model dirs, backend, style prompt |
| `history.json` | 识别历史（最多 500 条）History (max 500) |
| `shuo.log` | 运行日志（UTF-8）Runtime log (UTF-8) |
| `models/` | 一键下载的模型落盘处 Downloaded models live here |

- 默认快捷键：鼠标侧键（后退），可在左上角「快捷键」按钮自定义
- 右上角下拉框切换界面语言（27 种）

## 依赖 / Dependencies

全 Rust（静态链接进可执行文件，安装包不依赖 Python / Qt）：

| 库 Library | 用途 Purpose |
|---|---|
| tauri 3 | 应用框架 App framework |
| llama-cpp-sys-2 | llama.cpp 绑定（GGUF 推理，主后端）llama.cpp bindings (GGUF, primary) |
| ort | ONNX Runtime 绑定（0.6B 后备后端 + 降噪）ONNX Runtime (fallback ASR + denoise) |
| cpal | 音频录制 Audio capture |
| tokenizers | 文本分词（ONNX 后端）Tokenization (ONNX path) |
| ureq | 模型下载（GitHub Release；TLS 走系统证书库）Model download (system trust store) |

## 许可证 / License

代码 Code：GPL-3.0

### 模型 / Models

模型权重版权归原作者所有，详见 [NOTICE](./NOTICE)。

Model weights: copyright belongs to original authors. See [NOTICE](./NOTICE).

识别模型 ASR model：[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)（Apache 2.0；GGUF 转换版见 ggml-org）
