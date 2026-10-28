# 说 · Shuo

Local speech recognition GUI on a **fine-tuned Qwen3-ASR** (llama.cpp/GGUF, 1.7B): transcription with **prompt-controlled output style**. This repo is an **app that calls the model** — the fine-tuning pipeline lives outside the repo; the fine-tuned model package is hosted on **ModelScope** (`a2heng/shuo-asr-1.7b-gguf`; model author: **OrientCOMPASS**, this repo only hosts/distributes it) and can be **downloaded in-app with one click**.

Single implementation: Tauri 3 (`src-tauri/`), all-Rust inference (`llama-cpp-sys-2` primary + `ort` fallback/denoise), static HTML/CSS/JS UI, no Python at runtime. State lives in `~/.shuo/` (config.json / history.json / shuo.log / models/).

## Commands

| Action | Command |
|--------|---------|
| Run | `cd src-tauri && cargo run --release` |
| Build (Linux) | `cd src-tauri && cargo tauri build --bundles deb appimage`（llama.cpp 静态链接需要 clang/libclang） |
| Build (Windows) | `powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1` |
| Rust tests | `cd src-tauri && cargo test --release`（真实模型用例由 `SHUO_GGUF_LM`+`SHUO_GGUF_MM`+`SHUO_GGUF_WAV` 环境变量门控，缺失自动跳过；**联网用例**：`cargo test --release -- --ignored modelscope_assets` 校验 ModelScope 清单 + 下 `shuo-model.json` 比对 SHA256） |
| 发版（日期版本） | `./scripts/release.sh`（版本 = `YY.MM.DD`，tag `vYY.MM.DD`；同日重复加 `-2`）|
| 版本注入（CI） | `bash ./scripts/set-version.sh <版本>`（只由 CI 从 tag 调用；仓库里不提交版本改动）|
| 获取模型（应用内） | 设置 → 模型与引擎 → **下载模型**（ModelScope `a2heng/shuo-asr-1.7b-gguf` → `~/.shuo/models/Qwen3-ASR-1.7B-GGUF-tuned/`，断点续传 + SHA256，完成后自动重载引擎） |
| 获取模型（官方基座 GGUF） | `hf download ggml-org/Qwen3-ASR-1.7B-GGUF --include "Qwen3-ASR-1.7B-Q8_0.gguf" "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" --local-dir Qwen3-ASR-1.7B-GGUF` |

**China mirror:** Set `$env:HF_ENDPOINT = "https://hf-mirror.com"` before downloads.

## Architecture

| File | Role |
|------|------|
| `src/main.rs` | Entry, commands (`bootstrap`/`save_config`/`set_hotkey`/`start_record`/`insert_text`/`list_style_presets`/`download_model`/`model_download_info`/`open_model_page`…), tray, portal app-id registration |
| `src/asr.rs` | ASR 公共契约（**无推理代码**）：`AsrResult`、`style_preset_list()`/`style_preset_prompt()` 预设（与模型包 `shuo-model.json` 的 `prompts` 记录逐字对表）、测试用 `hound_lite` WAV 读取；底部 `prompt_contract` 用 cfg(test) 保留 Qwen3 chat 模板的 token 级布局作回归对照 |
| `src/gguf_asr.rs` | ASR pipeline (GGUF/llama.cpp，主路径): llama-cpp-sys-2 静态链接（openmp+mtmd），LM+mmproj 加载、mtmd 音频 tokenize/prefill、贪心解码、`language X<asr_text>` 解析；对外 `transcribe`/`transcribe_single` → `AsrResult`（唯一 ASR 后端） |
| `src/model_download.rs` | 模型一键下载：ModelScope `repo/files` 查清单（Size + Sha256）→ 流式落盘（.part + Range 断点续传 + 大小/SHA256 校验）→ `model-download` 进度事件 → 完成后 `reload_engine_full`。**下载源 = 顶部 `REPO`/`REVISION` 常量**（`a2heng/shuo-asr-1.7b-gguf` @ `master`，公开仓库免登录；`SHUO_MODEL_ENDPOINT` 可覆盖域名指向镜像）；TLS 走系统证书库（ureq `platform-verifier`，共享 Agent） |
| `src/mel.rs` | Mel filters/log-mel/VAD splitting/AGC — numerically equal to librosa (max diff 2e-5) |
| `src/denoise.rs` | Streaming ONNX denoiser (NFT_960, 48 kHz) |
| `src/audio.rs` | cpal always-on capture + 0.5 s pre-buffer + 500 ms release debounce |
| `src/hotkey.rs` | Spec parsing (keyboard + mouse side buttons), platform dispatch |
| `src/hotkey_windows.rs` | Low-level keyboard/mouse hooks (press **and** release, injected-event filter) |
| `src/hotkey_linux.rs` | GlobalShortcuts portal (Wayland, primary) + evdev fallback (X11 / mouse buttons) |
| `src/typing_windows.rs` | SendInput + `KEYEVENTF_UNICODE` |
| `src/typing_linux.rs` | uinput virtual keyboard for paste shortcuts (no portal prompt) → RemoteDesktop (persistent restore token, re-request on revoke) → wtype/ydotool/xdotool → clipboard |
| `src/config.rs` | `~/.shuo/config.json` + history; 输出风格字段 `style_{preset,prompt}`（serde alias 兼容旧字段名）+ `Config::style_context()`；历史版本废弃字段（含已删的 `asr_backend`）经 `extra` flatten 原样保留 |
| `src/models.rs` | Model dir resolution（只有 GGUF）：槽位 `Qwen3-ASR-1.7B-GGUF[-tuned]/`（mmproj 前缀=音频编码器，最大 .gguf=LM）；微调包以 `shuo-model.json` 标记；`resolve()`/`describe()` |
| `src/state.rs` | AppState, loader/worker threads, events `engine-status`/`rec`/`asr`/`final`/`hotkey`/`model-download`; ASR 槽位 `Mutex<Option<GgufAsr>>`（单一 llama.cpp 后端，加载失败直接报错）；context（风格提示词）恒随识别请求进 system 段，识别结果即最终文本，`final` 事件带 `style_model` 标志；`reload_engine_full()` 重载路径 |
| `ui/` | Static frontend (index.html / style.css / app.js / locales)；设置抽屉含「输出风格」区（预设下拉/自定义提示词/模型包状态行）与「模型下载」区（一键下载 + 进度条）；状态栏 ASR chip 显示后端（恒 GGUF）；最终卡片角标显示生效的风格预设名 |
| `linux/70-shuo-hotkey.rules` | udev rules for evdev + uinput (`/dev/uinput` uaccess), installed by deb |
| `linux/shuo-overlay.py` | XWayland status capsule helper (bottom, always-on-top, no focus); bundled as resource |

## Key facts

- **Model dirs in .gitignore:** `Qwen3-ASR-1.7B-GGUF[-tuned]/`（`Qwen3-ASR-0.6B-ONNX-CPU/`、`Qwen3.5-2B-ONNX-OPT/` 仍在 ignore 里，但已是历史目录，应用不再读取）— 应用内一键下载，或手动放置
- **Model lookup:** config path → `SHUO_MODEL_DIR` → `~/.shuo/models/` → exe ancestors → cwd ancestors → repo dir → `/usr/lib/Shuo/models`；GGUF 先试 `Qwen3-ASR-1.7B-GGUF-tuned/` 再试 `Qwen3-ASR-1.7B-GGUF/`
- **ASR 后端：** 只有 llama.cpp/GGUF（CPU，静态链接，无 GPU 依赖）；ONNX ASR 后备后端已于 2026-10-28 移除，`ort` 只服务降噪；`Status.asr_backend` 恒 `"gguf"`（UI chip 显示用），配置里的 `asr_backend` 已删除（旧值保留在 `extra`）
- **性能 / 线程数：** 推理线程 = **物理核心数**（`num_cpus::get_physical()`，上限 16）。超线程会让 llama.cpp 严重变慢（Ryzen 5 7500F 实测 10.8 s 音频：6 线程 2.4 s / 8 线程 2.6 s / 10 线程 4.8 s / 12 线程 19–40 s）。`SHUO_ASR_THREADS` / `SHUO_MTMD_THREADS` 可覆盖；mtmd 默认 n_threads=4，已改为跟随。
- **性能参考值（本机 6 线程，10.4 s 音频）：** 降噪 0.5 s + 音频编码 0.5 s + prefill/解码 1.2–1.5 s ≈ **2.2–2.5 s（RTF 0.22）**；每次识别写 `识别完成：音频/降噪/ASR/总/RTF` 日志。
- **性能基准（手动跑，不进 CI）：** `cargo test --release gguf_transcribe_bench -- --ignored --nocapture`（`SHUO_GGUF_LM`/`SHUO_GGUF_MM`/`SHUO_GGUF_WAV`、可选 `SHUO_ASR_THREADS`/`SHUO_MTMD_THREADS`）；`denoise_bench` 只需 `SHUO_GGUF_WAV`。
- **AVX-512（可选，仅本机）：** `GGML_AVX512=ON GGML_AVX512_VNNI=ON GGML_AVX512_BF16=ON GGML_AVX512_VBMI=ON cargo build --release`；注意 llama-cpp-sys-2 的 cmake cache 会粘住旧值，切回可移植构建要先删 `target/release/build/llama-cpp-sys-2-*` 再显式 `GGML_AVX512=OFF …`。Zen 4 实测无收益。
- **llama.cpp 构建依赖：** cmake + C/C++ 编译器 + libclang（bindgen）；Linux 运行期依赖 libgomp1（deb depends 已加）；Windows 运行期依赖 VC++ 2015-2022 运行库（vcomp140，OpenMP）
- **风格提示词契约**：提示词 = `asr.rs::style_preset_prompt()` ↔ 模型包 `shuo-model.json` 的 `prompts` 记录**逐字一致**；进 chat 模板 system 段（context 通道），基座只当识别偏置，微调包才执行指令；`gguf_asr.rs` 的文本模板与训练端 `chat_template.jinja` 必须同构（`asr.rs` 的 `prompt_contract` 测试保留 token 级对照）
- **模型下载源**：ModelScope 公开仓库 `a2heng/shuo-asr-1.7b-gguf`（`model_download.rs` 的 `REPO`/`REVISION` 常量，`master` 分支）；资产名固定 `tuned-Q4_K_M.gguf` / `mmproj-tuned-q8.gguf` / `shuo-model.json`。
  - 为什么不用 GitHub Release / HF：实测**ModelScope 直连 ≈9.8 MB/s**，GitHub Release 直连 ≈30 KB/s（多连接 aria2 走代理才快），HuggingFace 需代理（直连超时），`hf-mirror.com` **只读不能上传**（登录/写接口 404）。
  - **模型作者是 [OrientCOMPASS](https://github.com/OrientCOMPASS)**（PR #1：数据 / LoRA 训练 / GGUF 量化）；a2heng 只做托管分发，**不是模型作者**。贡献者 fork `OrientCOMPASS/shuo` 的 pre-release `model-tuned-1.7B-pre` 仅作过渡，应用已不再引用。
  - 上传/更新模型：`modelscope` CLI/SDK（`~/.modelscope/credentials`；token 走 `MS_TOKEN` 环境变量）→ `api.upload_file(repo, "model", 本地文件, 仓库内路径)`；上传实测 ≈6 MB/s。
- **CI 布局**：`ci.yml`（push 必跑：rust ubuntu/windows 构建单测 + **测试构建 artifact `test-build-windows`/`test-build-linux`**（免安装直跑、含 denoise.onnx 与说明，实机测试首选）+ UI 语法与控制断言（含「无 finetune/legacy/i18n/LLM/旧话术/双卡片」防回归））；`release.yml`（tag 发版 / 手动 dispatch 构建 NSIS/deb/AppImage 安装包，全量无缓存较慢）
- **Config at `~/.shuo/config.json`**: hotkey/hotkey_terminal (+ _enabled), language, ASR lang, denoise, auto_type, mic_device, style_{preset,prompt}
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
