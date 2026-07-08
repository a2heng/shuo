# 说 · Shuo

Local speech recognition GUI: Qwen3-ASR (ONNX CPU) + Qwen3.5 text post-processing. Windows-only, no GPU/PyTorch.

## Commands

| Action | Command |
|--------|---------|
| Run | `python asr_gui.py` |
| Install deps | `pip install -r requirements.txt` |
| Download LLM | `hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT` |
| Download ASR model | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) | 
| Test ASR | `python test_asr.py` |
| Test LLM | `python test_llm.py` |
| Build installer | `.\build.ps1` (requires 7-Zip at `C:\Program Files\7-Zip\7z.exe`) |
| i18n extract | `python -m i18n.build_i18n_locales extract` (scans `.py` files for `i18n.tr(...)`) |
| i18n generate | `python -m i18n.build_i18n_locales generate` (regenerates all 27 locale files) |

**China mirror:** Set `$env:HF_ENDPOINT = "https://hf-mirror.com"` before downloads.

## Architecture

| File | Role |
|------|------|
| `onnx_infer/asr.py` | ASR pipeline (0.6B): mel spectrogram → ONNX encoder → decoder init/step → text |
| `onnx_infer/denoise.py` | Streaming ONNX denoiser (NFT_960, 48kHz input) |
| `onnx_infer/denoise.onnx` | Denoiser model file |
| `onnx_infer/llm.py` | Qwen3.5-2B ONNX post-processing (punctuation, de-um, correction) |
| `asr_gui.py` | Entry point — PySide6 GUI, audio recording, hotkey, config, result queue |
| `hotkey.py` | pynput global hotkey listener, persisted to `~/.shuo/config.json` |
| `i18n/__init__.py` | Minimal i18n: `i18n.load("zh")`, `i18n.tr("key")`, English fallback |
| `i18n/build_i18n_locales.py` | Scans `.py` for translation keys + generates all 27 locale files |

## Key facts

- **Model dirs in .gitignore:** `Qwen3-ASR-0.6B-ONNX-CPU/`, `Qwen3.5-*-ONNX-OPT/` — must be downloaded separately
- **Config at `~/.shuo/config.json`**: hotkey, language, ASR lang, denoise, auto_type, mic_device, LLM settings
- **Log at `~/.shuo/shuo.log`**: UTF-8, `logger = logging.getLogger("shuo")`
- **History at `~/.shuo/history.json`**: max 500 entries, newest first
- **PyAudio MME only:** GUI filters to MME devices; Chinese Windows may return GBK-encoded device names (fixed in app)
- **ASR output regex:** `^language\s+(\w+)\s*<asr_text>(.*)` in `onnx_infer/asr.py:147`
- **Recording:** Pre-buffer 0.5s rolling buffer, AGC peak-normalizes to -1 dBFS; release debounce 500ms
- **Auto-type:** SendInput + KEYEVENTF_UNICODE (`send_unicode_text()` at module level in `asr_gui.py:52`); no clipboard, works in terminals/Vim; non-BMP via UTF-16 surrogate pairs
- **LLM presets:** standard / moderate / aggressive / aggressive_no_punc / translate (in `onnx_infer/llm.py:18`)
- **Build:** PyInstaller via `build.ps1`; hooks in `hooks/` block torch/triton at collection time; post-build cleanup removes Tcl/Tk, Qt6Quick/Qml/Pdf, OpenGL SW dlls
- **Theme:** Qt system color scheme detection + Win32 `WM_SETTINGCHANGE` listener; dark/light with explicit color dicts
- **No test framework:** standalone `.py` test scripts (not pytest)
- **No CI/CD:** no `.github/` directory
- **Audio denoiser model** at `onnx_infer/denoise.onnx` in repo root (not gitignored)

## LLM threading

`onnx_infer/llm._inference_lock` serializes all `call_llm()` invocations. The model singleton loads lazily at first call. Cache prefix embeddings per system prompt — changing prompt triggers re-cache.

## i18n key patterns

- `settings.mic`, `settings.mic_default`, `settings.asr_lang_label`, `settings.asr_lang_tip`, `settings.ui_lang`
- `asr_lang.auto`, `asr_lang.zh`, ..., `asr_lang.sv` (27 languages)
- Dynamic keys in `i18n/build_i18n_locales.py:8-17` (in `cmd_extract()` function) — agent must add new ones there if introducing runtime-dynamic keys

## Build output

`build.ps1` produces:
- `dist/Shuo_yyyy-MM-dd-HHmm.exe` (full installer with models, 7z SFX)
- `dist/Shuo_nomodel_yyyy-MM-dd-HHmm.exe` (without model dirs, for upgrade use)



