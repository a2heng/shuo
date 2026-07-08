# Windows

[Back](../README.md)

## Install

### 1. Download ASR Model

> **China users:** Set mirror before download: `$env:HF_ENDPOINT = "https://hf-mirror.com"`

Clone Qwen3-ASR-0.6B-ONNX-CPU: `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone to project root)

### 2. Install Dependencies & LLM Model

> **China users:** Also set mirror before downloading: `$env:HF_ENDPOINT = "https://hf-mirror.com"`

```powershell
pip install -r requirements.txt
hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT
```

### 3. Launch

```powershell
python asr_gui.py
```

## Deploy

### Prerequisites

- [PyInstaller](https://pyinstaller.org/): `pip install pyinstaller`
- [7-Zip](https://7-zip.org/) (for packaging the self-extracting installer)

### Build Single-File Installer

```powershell
.\build.ps1
```

Output: `dist/Shuo_yyyy-MM-dd-HHmm.exe` (self-extracting installer)
