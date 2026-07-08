## Qwen3-ASR-0.6B — Full ONNX CPU

> **Run Qwen3-ASR on any CPU. No GPU. No PyTorch.**

Self-contained ONNX pipeline for [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Encoder and decoder both run on ONNX Runtime with INT8 quantized decoder. Long audio is automatically split at silence boundaries — no manual chunking needed.

## Highlights

-   **Zero GPU, Zero PyTorch** — `onnxruntime` + `librosa` + `tokenizers` only
-   **Real-time on 8W CPU** — RTF 0.71x with VAD chunking on Intel N100
-   **3x realtime on desktop** — RTF 0.32x (INT8)
-   **Self-contained** — all weights + tokenizer included
-   **30 languages** — same multilingual coverage as the original
-   **Long audio support** — automatic silence-based splitting with `--chunk-sec`
-   **Bit-exact encoder** — cosine similarity 1.000000 vs PyTorch reference

## Quick Start

```bash
git clone https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU
cd Qwen3-ASR-0.6B-ONNX-CPU

python3 -m venv .venv && source .venv/bin/activate
pip install onnxruntime librosa soundfile tokenizers

# Short audio
python onnx_inference.py test_audio/librispeech_1_1089_1.wav

# Long audio (auto-chunked at silence)
python onnx_inference.py long_meeting.wav --chunk-sec 30

# Specify language + JSON output
python onnx_inference.py audio.wav --language Korean --json
```

## Benchmarks

### Intel N100 (4 cores, 8W TDP)

**Short audio** — 13 LibriSpeech test-clean samples, INT8 decoder:

| Audio | RTF | Encoder | Prefill | Decode | Tokens |
| --- | --- | --- | --- | --- | --- |
| 12.4s | **0.97x** | 3.8s | 3.2s | 5.1s | 49 |
| 11.6s | **0.84x** | 2.4s | 3.0s | 4.3s | 39 |
| 10.6s | **0.86x** | 2.3s | 2.9s | 3.8s | 32 |
| 10.4s | **0.92x** | 1.6s | 2.5s | 4.5s | 41 |
| 6.6s | 1.08x | 1.5s | 2.3s | 3.4s | 27 |
| 3.3s | 1.36x | 1.3s | 1.4s | 1.7s | 15 |

> Audio > 10s achieves realtime or faster. Decoder: ~100ms/token (INT8).

**Long audio** — production deployment with VAD chunking (Docker, 2 threads):

| Input | Chunks | Avg RTF | Peak Memory |
| --- | --- | --- | --- |
| 600s (10 min) | 19 | **0.71x** | 5.7 GB |

> Without chunking, 10-min audio consumes 15GB+ and gets OOM-killed. With 30s chunks, prefill cost is amortized → RTF drops well below 1.0x.

**Desktop x86\_64:**

| Mode | RTF | Notes |
| --- | --- | --- |
| ONNX FP32 | 0.63x | No quantization |
| **ONNX INT8** | **0.32x** | 3x realtime |

## Architecture

| Stage | Runtime | Component | Details |
| --- | --- | --- | --- |
| 1 | librosa | Mel Spectrogram | 16kHz → 128-bin log-mel |
| 2 | ONNX Runtime | `encoder_conv.onnx` | 3x Conv2D, 8x downsample |
| 3 | ONNX Runtime | `encoder_transformer.onnx` | 18 Transformer layers + Projector (896→1024) |
| 4 | NumPy | `embed_tokens.bin` | Fuse audio features into prompt |
| 5 | ONNX Runtime | `decoder_init.int8.onnx` | Prefill → logits + KV cache |
| 6 | ONNX Runtime | `decoder_step.int8.onnx` | Autoregressive decode until EOS |

|  | Encoder | Decoder |
| --- | --- | --- |
| Quantization | FP32 | Dynamic INT8 |
| Format | 2 models (conv + transformer) | 2 models (init + step) |
| KV Cache | — | ONNX I/O |

## Long Audio

Audio longer than 45s is automatically split at silence boundaries using RMS energy detection. No external VAD model needed.

```bash
python onnx_inference.py meeting.wav                  # 30s chunks (default)
python onnx_inference.py meeting.wav --chunk-sec 20   # 20s chunks, less memory
```

Split range scales with target: min = target/2, max = target×1.5. The split point is the silence frame nearest to the target length.

## Files

| File | Size | Description |
| --- | --- | --- |
| `onnx_inference.py` | — | Inference CLI (single file, no deps beyond pip) |
| `tokenizer.json` | 11 MB | Self-contained tokenizer |
| `onnx_models/encoder_conv.onnx` | 48 MB | Conv block |
| `onnx_models/encoder_transformer.onnx` | 669 MB | Transformer + Projector |
| `onnx_models/decoder_init.int8.onnx` | 571 MB | Prefill (INT8) |
| `onnx_models/decoder_step.int8.onnx` | 571 MB | Decode step (INT8) |
| `onnx_models/embed_tokens.bin` | 622 MB | Token embeddings |

**Total: ~2.5 GB**

## Model

| Component | Params | Details |
| --- | --- | --- |
| Audio Encoder | ~310M | d=896, 18 layers, 14 heads |
| Projector | ~1.7M | Linear 896→1024 |
| LLM Decoder | ~470M | d=1024, 28 layers, GQA 16Q/8KV |
| **Total** | **~782M** |  |

**30 languages:** Chinese, English, Cantonese, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Russian, Thai, Vietnamese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

## Technical Notes

-   **Attention**: Original `cu_seqlens` windowed attention only works with `flash_attention_2`. CPU uses eager mode (all-to-all). ONNX export matches this.
-   **Weight tying**: `embed_tokens` = `lm_head` in original. In ONNX, separated — `embed_tokens.bin` for input, `lm_head` baked into decoder.
-   **KV Cache**: `[num_layers, batch, kv_heads, seq_len, head_dim]` — init outputs, step extends.
-   **MRoPE**: Layout \[24,20,20\]. For ASR (no vision), all 3 dims share the same position IDs.

## Dependencies

```
onnxruntime
librosa
soundfile
tokenizers
```

No PyTorch. No transformers. No CUDA.

## Acknowledgements

Decoder ONNX export architecture inspired by [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx).

## References

-   [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — Original model
-   [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR) — Official repo
-   [Technical Report](https://arxiv.org/abs/2601.21337) — arXiv:2601.21337
-   [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx) — ONNX export reference

## License

Code: Apache 2.0. Model weights: [original license](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).