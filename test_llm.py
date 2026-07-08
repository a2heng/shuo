#!/usr/bin/env python3
"""Test Qwen3.5-2B-ONNX-OPT — pure ONNX CPU inference (no GPU, no PyTorch)."""

import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer

MODEL_DIR = Path(__file__).parent / "Qwen3.5-2B-ONNX-OPT"
ONNX_DIR = MODEL_DIR / "onnx"

EOS_ID = 248044
EOS_ID_ALT = 248046

SYSTEM_PROMPT = "You are a helpful assistant."

TEST_CASES = [
    ("简单问答", "你好，请问你是谁？"),
    ("标点优化", "请优化标点符号：今天天气真好啊我们一起去公园玩吧"),
    ("去口癖", "请去掉口吃：然后那个就是我觉得这个东西还挺好的然后就是想问一下能不能用"),
]


class Qwen3_5Onnx:
    def __init__(self, model_dir: str = str(MODEL_DIR)):
        model_dir = Path(model_dir)
        onnx_dir = model_dir / "onnx"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 3

        self.embed = ort.InferenceSession(str(onnx_dir / "embed_tokens_q4f16.onnx"), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(onnx_dir / "decoder_model_merged_q4f16.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        self.decoder_out_names = [o.name for o in self.decoder.get_outputs()]

    def _empty_past(self, batch=1):
        past = {}
        for i in range(24):
            if i % 4 == 3:
                past[f"past_key_values.{i}.key"] = np.zeros((batch, 2, 0, 256), dtype=np.float16)
                past[f"past_key_values.{i}.value"] = np.zeros((batch, 2, 0, 256), dtype=np.float16)
            else:
                past[f"past_conv.{i}"] = np.zeros((batch, 6144, 3), dtype=np.float16)
                past[f"past_recurrent.{i}"] = np.zeros((batch, 16, 128, 128), dtype=np.float16)
        return past

    def _present_to_past(self, outputs):
        past = {}
        for name, arr in zip(self.decoder_out_names, outputs):
            if name.startswith("present_"):
                past[name.replace("present_", "past_", 1)] = arr
            elif name.startswith("present."):
                past[name.replace("present.", "past_key_values.", 1)] = arr
        return past

    def _apply_template(self, text: str) -> str:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def generate(self, text: str, max_new: int = 128, temperature: float = 0.7):
        prompt = self._apply_template(text)
        tokens = self.tokenizer.encode(prompt).ids
        seq_len = len(tokens)

        t0 = time.time()

        # Embed
        emb = self.embed.run(None, {"input_ids": np.array([tokens], dtype=np.int64)})[0]

        # Prefill
        mask = np.ones((1, seq_len), dtype=np.int64)
        pos = np.arange(seq_len, dtype=np.int64)
        pos_ids = np.stack([pos, pos, pos], axis=0)[:, np.newaxis, :]

        feed = {"inputs_embeds": emb, "attention_mask": mask, "position_ids": pos_ids,
                "num_logits_to_keep": np.array(1, dtype=np.int64)}
        feed.update(self._empty_past())
        outputs = self.decoder.run(None, feed)
        logits = outputs[0]
        state = self._present_to_past(outputs)

        next_id = int(np.argmax(logits[0, -1]))
        ids = [next_id]
        cur_pos = seq_len

        for _ in range(max_new - 1):
            if next_id in (EOS_ID, EOS_ID_ALT):
                break
            emb = self.embed.run(None, {"input_ids": np.array([[next_id]], dtype=np.int64)})[0]
            mask = np.ones((1, cur_pos + 1), dtype=np.int64)
            pos_ids = np.full((3, 1, 1), cur_pos, dtype=np.int64)

            feed = {"inputs_embeds": emb, "attention_mask": mask, "position_ids": pos_ids,
                    "num_logits_to_keep": np.array(1, dtype=np.int64)}
            feed.update(state)
            outputs = self.decoder.run(None, feed)
            logits = outputs[0]
            state = self._present_to_past(outputs)

            next_id = int(np.argmax(logits[0, -1]))
            ids.append(next_id)
            cur_pos += 1

        elapsed = time.time() - t0
        out = self.tokenizer.decode(ids, skip_special_tokens=True)
        return out, elapsed, len(ids)


def main():
    model = Qwen3_5Onnx()

    print("=" * 60)
    print("  Qwen3.5-2B-ONNX-OPT — 纯 CPU 推理测试")
    print("=" * 60)
    print()

    for name, text in TEST_CASES:
        print("-" * 60)
        print(f"  [{name}]  输入: {text}")
        result, elapsed, tokens = model.generate(text, max_new=128)
        print(f"  耗时: {elapsed:.2f}s  tokens: {tokens}  ({tokens/elapsed:.1f} tok/s)")
        print(f"  输出: {result or '(空)'}")
        print()

    print("=" * 60)
    print("  完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
