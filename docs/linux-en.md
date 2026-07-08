# Linux

[Back](../README.md)

## Install

### 1. Download ASR Model

> **China users:** Set mirror before download: `export HF_ENDPOINT=https://hf-mirror.com`

Clone Qwen3-ASR-0.6B-ONNX-CPU: `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone to project root)

### 2. Install Dependencies & LLM Model

> **China users:** Also set mirror before downloading: `export HF_ENDPOINT=https://hf-mirror.com`

```bash
pip install -r requirements.txt
hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT
```

### 3. Launch

```bash
python asr_gui.py
```

## Deploy

```bash
pip install pyinstaller
pyinstaller --windowed --name Shuo --icon=shuo.ico -y \
  --add-data "locales:locales" \
  --add-data "Qwen3-ASR-0.6B-ONNX-CPU:Qwen3-ASR-0.6B-ONNX-CPU" \
  --add-data "Qwen3.5-2B-ONNX-OPT:Qwen3.5-2B-ONNX-OPT" \
  --hidden-import onnxruntime --hidden-import tokenizers \
  --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py
```

Output: `dist/Shuo/`
