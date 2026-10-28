# 说 · Shuo

Local speech recognition GUI on a **fine-tuned Qwen3-ASR** (llama.cpp/GGUF, 1.7B): transcription with **prompt-controlled output style**. This repo is an **app that calls the model** — the fine-tuning pipeline lives outside the repo; the fine-tuned model package is distributed via GitHub Release and can be **downloaded in-app with one click**.

Single implementation: Tauri 3 (`src-tauri/`), all-Rust inference (`llama-cpp-sys-2` primary + `ort` fallback/denoise), static HTML/CSS/JS UI, no Python at runtime. State lives in `~/.shuo/` (config.json / history.json / shuo.log / models/).

## Commands

| Action | Command |
|--------|---------|
| Run | `cd src-tauri && cargo run --release` |
| Build (Linux) | `cd src-tauri && cargo tauri build --bundles deb appimage`（llama.cpp 静态链接需要 clang/libclang） |
| Build (Windows) | `powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1` |
| Rust tests | `cd src-tauri && cargo test --release`（真实模型用例由 `SHUO_ASR_DIR` / `SHUO_GGUF_LM`+`SHUO_GGUF_MM`+`SHUO_GGUF_WAV` 环境变量门控，缺失自动跳过） |
| 发版（日期版本） | `./scripts/release.sh`（版本 = `YY.MM.DD`，tag `vYY.MM.DD`；同日重复加 `-2`）|
| 版本注入（CI） | `bash ./scripts/set-version.sh <版本>`（只由 CI 从 tag 调用；仓库里不提交版本改动）|
| 获取模型（应用内） | 设置 → 模型与引擎 → **下载模型**（GitHub Release → `~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/`，断点续传，完成后自动重载引擎） |
| 获取模型（官方基座 GGUF） | `hf download ggml-org/Qwen3-ASR-1.7B-GGUF --include "Qwen3-ASR-1.7B-Q8_0.gguf" "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" --local-dir Qwen3-ASR-1.7B-GGUF` |
| 获取模型（ONNX 后备） | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) |

**China mirror:** Set `$env:HF_ENDPOINT = "https://hf-mirror.com"` before downloads.

## Architecture

| File | Role |
|------|------|
| `src/main.rs` | Entry, commands (`bootstrap`/`save_config`/`set_hotkey`/`start_record`/`insert_text`/`list_style_presets`/`download_model`/`model_download_info`/`open_release_page`…), tray, portal app-id registration |
| `src/asr.rs` | ASR pipeline (ONNX/ort): mel → conv/transformer encoders → decoder init/step → tokens; `prompt_ids()` 纯函数构造 chat 模板（system 段带 context=输出风格提示词）；`style_preset_list()`/`style_preset_prompt()` 预设（与模型包 `shuo-model.json` 的 `prompts` 记录逐字对表） |
| `src/gguf_asr.rs` | ASR pipeline (GGUF/llama.cpp，主路径): llama-cpp-sys-2 静态链接（openmp+mtmd），LM+mmproj 加载、mtmd 音频 tokenize/prefill、贪心解码、`language X<asr_text>` 解析；与 AsrPipeline 同接口（transcribe/transcribe_single → AsrResult） |
| `src/model_download.rs` | 模型一键下载：GitHub Release API 查资产 → 流式落盘（.part + Range 断点续传 + 大小校验）→ `model-download` 进度事件 → 完成后 `reload_engine_full`。**下载源是顶部两个常量 `RELEASE_REPO`/`RELEASE_TAG`**（当前指向贡献者 fork 的 pre-release，上游合并后改成自己的 Release 即可）；TLS 走系统证书库（ureq `platform-verifier`，共享 Agent） |
| `src/mel.rs` | Mel filters/log-mel/VAD splitting/AGC — numerically equal to librosa (max diff 2e-5) |
| `src/denoise.rs` | Streaming ONNX denoiser (NFT_960, 48 kHz) |
| `src/audio.rs` | cpal always-on capture + 0.5 s pre-buffer + 500 ms release debounce |
| `src/hotkey.rs` | Spec parsing (keyboard + mouse side buttons), platform dispatch |
| `src/hotkey_windows.rs` | Low-level keyboard/mouse hooks (press **and** release, injected-event filter) |
| `src/hotkey_linux.rs` | GlobalShortcuts portal (Wayland, primary) + evdev fallback (X11 / mouse buttons) |
| `src/typing_windows.rs` | SendInput + `KEYEVENTF_UNICODE` |
| `src/typing_linux.rs` | uinput virtual keyboard for paste shortcuts (no portal prompt) → RemoteDesktop (persistent restore token, re-request on revoke) → wtype/ydotool/xdotool → clipboard |
| `src/config.rs` | `~/.shuo/config.json` + history; 输出风格字段 `style_{preset,prompt}`（serde alias 兼容旧字段名）+ `asr_backend`（auto/onnx/gguf）+ `Config::style_context()`；历史版本废弃字段经 `extra` flatten 原样保留 |
| `src/models.rs` | Model dir resolution; GGUF 槽位 `Qwen3-ASR-1.7B-GGUF[-tuned]/`（mmproj 前缀=音频编码器，最大 .gguf=LM）；ONNX 槽位优先 `Qwen3-ASR-0.6B-ONNX-CPU-tuned/`；微调包以 `shuo-model.json` 标记；`pick_backend()`/`tuned_for()` |
| `src/state.rs` | AppState, loader/worker threads, events `engine-status`/`rec`/`asr`/`final`/`hotkey`/`model-download`; `AsrEngine` 双后端（Gguf 主/Onnx 后备，加载失败自动回退）；context（风格提示词）恒随识别请求进 system 段，识别结果即最终文本，`final` 事件带 `style_model` 标志；`reload_engine_full()` 重载路径 |
| `ui/` | Static frontend (index.html / style.css / app.js / locales)；设置抽屉含「输出风格」区（预设下拉/自定义提示词/模型包状态行）与「模型下载」区（一键下载 + 进度条）；状态栏 ASR chip 显示后端（GGUF/ONNX）；最终卡片角标显示生效的风格预设名 |
| `linux/70-shuo-hotkey.rules` | udev rules for evdev + uinput (`/dev/uinput` uaccess), installed by deb |
| `linux/shuo-overlay.py` | XWayland status capsule helper (bottom, always-on-top, no focus); bundled as resource |

## Key facts

- **Model dirs in .gitignore:** `Qwen3-ASR-1.7B-GGUF[-tuned]/`, `Qwen3-ASR-0.6B-ONNX-CPU[-tuned]/` — 应用内一键下载，或手动放置
- **Model lookup:** config path → `SHUO_MODEL_DIR` → `~/.shuo/models/` → exe ancestors → cwd ancestors → repo dir → `/usr/lib/Shuo/models`；GGUF 先试 `Qwen3-ASR-1.7B-GGUF-tuned/` 再试 `Qwen3-ASR-1.7B-GGUF/`；ONNX 先试 `Qwen3-ASR-0.6B-ONNX-CPU-tuned/` 再试基座包名
- **ASR 后端选择：** `asr_backend` 配置 auto/onnx/gguf；auto=有 GGUF 模型优先 GGUF（1.7B 主路径）；显式后端加载失败自动回退另一个（日志说明）；`Status.asr_backend` 上报 UI
- **llama.cpp 构建依赖：** cmake + C/C++ 编译器 + libclang（bindgen）；Linux 运行期依赖 libgomp1（deb depends 已加）；Windows 运行期依赖 VC++ 2015-2022 运行库（vcomp140，OpenMP）
- **风格提示词契约**：提示词 = `asr.rs::style_preset_prompt()` ↔ 模型包 `shuo-model.json` 的 `prompts` 记录**逐字一致**；进 chat 模板 system 段（context 通道），基座只当识别偏置，微调包才执行指令；GGUF 后端同一契约（`gguf_asr.rs` 的 prompt 模板与 `asr.rs::prompt_ids` 逐 token 同构）
- **模型下载源**：`model_download.rs` 的 `RELEASE_REPO`/`RELEASE_TAG` 常量（当前 `OrientCOMPASS/shuo` @ `model-tuned-1.7B-pre`，过渡期托管）；资产名固定 `tuned-Q4_K_M.gguf` / `mmproj-tuned-q8.gguf` / `shuo-model.json`，上游换源只需改常量
- **CI 布局**：`ci.yml`（push 必跑：rust ubuntu/windows 构建单测 + **测试构建 artifact `test-build-windows`/`test-build-linux`**（免安装直跑、含 denoise.onnx 与说明，实机测试首选）+ UI 语法与控制断言（含「无 finetune/legacy/i18n/LLM/旧话术/双卡片」防回归））；`release.yml`（tag 发版 / 手动 dispatch 构建 NSIS/deb/AppImage 安装包，全量无缓存较慢）
- **Config at `~/.shuo/config.json`**: hotkey/hotkey_terminal (+ _enabled), language, ASR lang, denoise, auto_type, mic_device, asr_backend, style_{preset,prompt}
- **Log at `~/.shuo/shuo.log`**: UTF-8, `log_line()` (Rust)
- **History at `~/.shuo/history.json`**: max 500 entries, newest first
- **Recording:** Pre-buffer 0.5s rolling buffer, AGC peak-normalizes to -1 dBFS; release debounce 500ms
- **Auto-type:** Windows SendInput + KEYEVENTF_UNICODE; Linux clipboard + paste shortcut via uinput (primary) or RemoteDesktop portal (fallback)
- **Global hotkey:** Windows = low-level hooks (RegisterHotKey can't detect release); Linux = `GlobalShortcuts` portal (needs host app-id registration + `com.shuo.desktop.desktop`), evdev as fallback
- **Inference threading:** 单一推理 worker 线程串行处理识别任务（context 恒注入，无第二段模型）；下载在独立线程，完成后原子替换引擎槽位
- **Theme:** `prefers-color-scheme` (web UI)
- **Audio denoiser model** at `src-tauri/resources/denoise.onnx` (bundled)

## i18n

- UI 文案源 = `src-tauri/ui/locales/*.json`（27 语言 + index.json）；Tauri-only 新键先加 `src-tauri/ui/app.js` 的 `EXTRA`（zh/en，其它语言回退英文）
- CI `ui` job 校验 app.js 语法 + 全部 locales JSON 可解析 + 控制存在性断言

## Build output

- Linux: `src-tauri/target/release/bundle/{deb,appimage}/*`
- Windows: `bundle/nsis/Shuo_<version>_x64-setup.exe`
