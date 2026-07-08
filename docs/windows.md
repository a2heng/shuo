# Windows

[返回](../README.md)

## 安装

> **中国大陆用户：** 下载 ASR 模型前请先设置镜像 `$env:HF_ENDPOINT = "https://hf-mirror.com"`，否则无法访问 HuggingFace。

### 1. 下载 ASR 模型

克隆 Qwen3-ASR-0.6B-ONNX-CPU 模型：`hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU`（克隆到项目根目录）

### 2. 安装依赖 & LLM 模型

> **中国大陆用户：** 下载前同样请先设置镜像 `$env:HF_ENDPOINT = "https://hf-mirror.com"`。

```powershell
pip install -r requirements.txt
hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT
```

### 3. 启动

```powershell
python asr_gui.py
```

## 部署

### 前置依赖

- [PyInstaller](https://pyinstaller.org/)：`pip install pyinstaller`
- [7-Zip](https://7-zip.org/)（用于打包自解压安装包）

### 构建单文件安装包

```powershell
.\build.ps1
```

输出：`dist/Shuo_yyyy-MM-dd-HHmm.exe`（自解压安装包）
