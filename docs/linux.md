# Linux

[返回](../README.md)

## 安装

> **中国大陆用户：** 下载 ASR 模型前请先设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`，否则无法访问 HuggingFace。

### 1. 下载 ASR 模型

克隆 Qwen3-ASR-0.6B-ONNX-CPU 模型：`hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU`（克隆到项目根目录）

### 2. 安装依赖 & LLM 模型

> **中国大陆用户：** 下载前同样请先设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`。

```bash
pip install -r requirements.txt
hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT
```

### 3. 启动

```bash
python asr_gui.py
```

## 部署

```bash
pip install pyinstaller
pyinstaller --windowed --name Shuo --icon=shuo.ico -y \
  --add-data "locales:locales" \
  --add-data "Qwen3-ASR-0.6B-ONNX-CPU:Qwen3-ASR-0.6B-ONNX-CPU" \
  --add-data "Qwen3.5-2B-ONNX-OPT:Qwen3.5-2B-ONNX-OPT" \
  --hidden-import onnxruntime --hidden-import tokenizers \
  --exclude-module torch --exclude-module sklearn --exclude-module tensorflow asr_gui.py
```

输出：`dist/Shuo/`
