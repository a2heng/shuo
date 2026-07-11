import sys
import os
import json
import re
import time
import logging
import threading
from pathlib import Path

logger = logging.getLogger("shuo")

# ONNX model paths
_MODEL_DIR = Path(__file__).parent.parent / "Qwen3.5-2B-ONNX-OPT"

# EOS token IDs
EOS_ID = 248044
EOS_ID_ALT = 248046

PRESETS = {
    "standard": (
        "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n"
        "规则：\n"
        "- 输入可能已带有部分标点，也可能完全没有标点。仅根据语义补充缺少的必要标点（逗号、句号、问号、感叹号），已有标点检查并改正；\n"
        "- 不补全缺失的内容或改写用词；\n"
        "- 标点宁少勿多，不确定时不加。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n"
        "示例：\n"
        "输入：下周去北京的机票订了吗？酒店我看了一下有三家备选你倾向哪个。\n"
        "输出：下周去北京的机票订了吗？酒店我看了一下有三家备选，你倾向哪个？\n"
        "输入：昨晚加班到九点回到家已经十点半了吃完饭又处理了几封邮件然后十二点才睡。\n"
        "输出：昨晚加班到九点，回到家已经十点半了，吃完饭又处理了几封邮件，然后十二点才睡。\n\n"
        "请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    ),
    "moderate": (
        "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n"
        "规则：\n"
        "- 输入可能已带有部分标点，也可能没有。根据语义补充缺少的标点，已有标点检查并改正，不补全缺失的内容；\n"
        "- 删除口癖和犹豫词（如嗯、啊、呃、那个、这个、就是说、然后、对吧、反正、大概、好像、就是、的话、之类的、这样的、对不对等）；\n"
        "- 不替换任何实词，保持原意。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n"
        "示例：\n"
        "输入：嗯那个我觉得就是说这个方案好像不太行然后可能得改一下，呃我再想想有没有更好的办法呃那个我们的服务器大概可能下周二吧要维护你提前备份一下数据\n"
        "输出：我觉得这个方案不太行，可能得改一下，我再想想有没有更好的办法。我们的服务器下周二要维护，你提前备份一下数据。\n"
        "输入：啊对了我忘了说。昨天的会议纪要我发你了你先看一下有什么问题明天再讨论 这个文档我大概看了一下，然后觉得内容方面还要补充。你那边方便加一下吗\n"
        "输出：对了，我忘了说。昨天的会议纪要我发你了，你先看一下，有什么问题明天再讨论。我大概看了一下，觉得内容还要补充。你那边方便加一下吗？\n\n"
        "请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    ),
    "aggressive": (
        "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n"
        "规则：\n"
        "- 补充缺失标点。\n"
        "- 删除口癖和犹豫词嗯、啊、那个、就是说、然后、反正、大概、好像、就是、的话等。\n"
        "- 修正明显的同音或近音识别错误。\n"
        "- 将中文数字（包括幺、两、拐、洞等读法）转为阿拉伯数字并合并，金额只转数字（如三块五→3.5元）。\n"
        "- 将日期时间转为标准格式（如二零二六年→2026年，下午三点半→15:30）。\n"
        "- 将常见物理单位替换为符号（如米→m，平方米→m²，摄氏度→℃，千瓦→kW等）。\n"
        "- 将数学口头表达式转为书写形式（如x的平方→x²，根号→√，f x等于x平方加一→f(x)=x²+1）。\n"
        "- 将邮箱的at和点替换为@和.，中文数字合并（如一二三四五六@qq点com→123456@qq.com）。\n\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n"
        "示例：\n"
        "输入：嗯那个函数f x等于x平方加一，下周二下午三点维护，温度二十五度三。\n"
        "输出：函数f(x)=x²+1，下周二15:00维护，温度25.3℃。\n"
        "请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    ),
    "aggressive_no_punc": (
        "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n"
        "规则：\n"
        "- 句之间标点改为空格，句尾标点去除，符号系统保留。\n"
        "- 删除口癖和犹豫词（嗯、啊、那个、就是说、然后、反正、大概、好像、就是、的话等）。\n"
        "- 修正明显的同音或近音识别错误。\n"
        "- 将中文数字（包括幺、两、拐、洞等读法）转为阿拉伯数字并合并，金额只转数字（如三块五→3.5元）。\n"
        "- 将日期时间转为标准格式（如二零二六年→2026年，下午三点半→15:30）。\n"
        "- 将常见物理单位替换为符号（如米→m，平方米→m²，摄氏度→℃，千瓦→kW等）。\n"
        "- 将数学口头表达式转为书写形式（如x的平方→x²，根号→√，f x等于x平方加一→f(x)=x²+1）。\n"
        "- 将邮箱的at和点替换为@和.，中文数字合并（如一二三四五六@qq点com→123456@qq.com）。\n\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n"
        "示例：\n"
        "输入：嗯那个函数f x等于x平方加一，下周二下午三点维护，温度二十五度三。\n"
        "输出：函数f(x)=x²+1 下周二15:00维护 温度25.3℃\n\n"
        "请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    ),
    "translate": (
        "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n"
        "规则：\n"
        "- 将文本翻译成英文，保持原意和语气；\n"
        "- 如果原文已经是英文，则原样输出；\n"
        "- 文本中的中文数字先转为阿拉伯数字再翻译（如三点→3:00、一百二十→120）。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n"
        "示例：\n"
        "输入：早上好你今天怎么样。下午三点有个会议大概一个小时你参加吗\n"
        "输出：Good morning, how are you today? I have a meeting at 3 PM, about one hour. Will you join?\n"
        "输入：这个项目下周五截止。请把文件发到我的邮箱然后抄送给李经理\n"
        "输出：This project is due next Friday. Please send the file to my email and cc Manager Li.\n"
        "输入：帮我订一张明天上午十点到上海的火车票。我的电话是幺三九零零八六拐五二一\n"
        "输出：Please book a train ticket to Shanghai for tomorrow at 10 AM. My phone number is 13900867521.\n"
        "输入：房间温度二十五度空调功率两千瓦开两个小时\n"
        "输出：Room temperature is 25 degrees, air conditioner power is 2000W, run for 2 hours. \n\n"
        "请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    ),
}

CUSTOM_PROMPTS_PATH = Path.home() / ".shuo" / "custom_prompts.json"

# Model singleton
_model = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()  # serializes concurrent call_llm
_model_ready = threading.Event()
_initialized = False


class Qwen3_5Onnx:
    def __init__(self, model_dir: str = str(_MODEL_DIR)):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_dir = model_dir / "onnx"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 3

        self.embed = ort.InferenceSession(
            str(onnx_dir / "embed_tokens_q4f16.onnx"), opts,
            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(
            str(onnx_dir / "decoder_model_merged_q4f16.onnx"), opts,
            providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.decoder_out_names = [o.name for o in self.decoder.get_outputs()]
        self._load_chat_template(model_dir)

    def _empty_past(self, batch=1):
        import numpy as np
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
        import numpy as np
        past = {}
        for name, arr in zip(self.decoder_out_names, outputs):
            if name.startswith("present_"):
                past[name.replace("present_", "past_", 1)] = arr
            elif name.startswith("present."):
                past[name.replace("present.", "past_key_values.", 1)] = arr
        return past

    def _load_chat_template(self, model_dir):
        import jinja2
        template_path = model_dir / "chat_template.jinja"
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
            env = jinja2.Environment(
                loader=jinja2.BaseLoader(),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._chat_template = env.from_string(template_str)
        else:
            self._chat_template = None

    def apply_chat_template(self, system_prompt, user_prompt, enable_thinking=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if self._chat_template:
            return self._chat_template.render(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tools=None,
            )
        # Fallback: manual chat format
        BOS = "\u003cim_start\u003e"
        EOS = "\u003cim_end\u003e"
        text = ""
        if system_prompt:
            text = f"{BOS}system\n{system_prompt}{EOS}\n"
        text += f"{BOS}user\n{user_prompt}{EOS}\n{BOS}assistant\n"
        if not enable_thinking:
            text += "<think>\n\n</think>\n\n"
        return text

    def generate(self, text: str, max_new: int = 512,
                 temperature: float = 1.0, top_k: int = 20,
                 top_p: float = 1.0, min_p: float = 0.0,
                 presence_penalty: float = 2.0,
                 repetition_penalty: float = 1.0):
        import numpy as np

        tokens = self.tokenizer.encode(text).ids
        seq_len = len(tokens)
        if seq_len == 0:
            return ""

        emb = self.embed.run(None, {"input_ids": np.array([tokens], dtype=np.int64)})[0]
        mask = np.ones((1, seq_len), dtype=np.int64)
        pos = np.arange(seq_len, dtype=np.int64)
        pos_ids = np.stack([pos, pos, pos], axis=0)[:, np.newaxis, :]

        feed = {"inputs_embeds": emb, "attention_mask": mask, "position_ids": pos_ids,
                "num_logits_to_keep": np.array(1, dtype=np.int64)}
        feed.update(self._empty_past())
        outputs = self.decoder.run(None, feed)
        logits = outputs[0]
        state = self._present_to_past(outputs)

        next_id = self._sample_token(logits[0, -1], temperature, top_k, top_p, min_p,
                                      presence_penalty, repetition_penalty, [])
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

            next_id = self._sample_token(logits[0, -1], temperature, top_k, top_p, min_p,
                                          presence_penalty, repetition_penalty, ids)
            ids.append(next_id)
            cur_pos += 1

        # Filter out think block by token IDs (248068=<think>, 248069=</think>)
        THINK_START = 248068
        THINK_END = 248069
        # Find content after last 
        out_ids = ids
        last_end = -1
        for i, tid in enumerate(ids):
            if tid == THINK_END:
                last_end = i
        if last_end >= 0:
            out_ids = ids[last_end + 1:]
        # Also strip leading think token if present
        if out_ids and out_ids[0] == THINK_START:
            out_ids = out_ids[1:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    def _sample_token(self, logits, temperature, top_k, top_p, min_p,
                      presence_penalty, repetition_penalty, generated_ids):
        import numpy as np

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            prev_logits = logits.copy()
            for tid in set(generated_ids):
                if prev_logits[tid] > 0:
                    prev_logits[tid] /= repetition_penalty
                else:
                    prev_logits[tid] *= repetition_penalty
            logits = prev_logits

        # Apply presence penalty
        if presence_penalty != 0.0 and generated_ids:
            for tid in set(generated_ids):
                logits[tid] -= presence_penalty

        if temperature <= 0.0:
            return int(np.argmax(logits))

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < np.sort(logits)[-top_k]
            logits[indices_to_remove] = -np.inf

        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -np.inf

        # Min-p filtering
        if min_p > 0.0:
            max_logit = np.max(logits)
            min_logit_threshold = max_logit + np.log(min_p)
            logits[logits < min_logit_threshold] = -np.inf

        # Convert to probabilities and sample
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Check for valid probabilities
        if np.any(np.isnan(probs)) or np.sum(probs) == 0:
            return int(np.argmax(logits))

        return int(np.random.choice(len(probs), p=probs))


def is_llm_available():
    onnx_dir = _MODEL_DIR / "onnx"
    return (onnx_dir / "embed_tokens_q4f16.onnx").exists() and \
           (onnx_dir / "embed_tokens_q4f16.onnx_data").exists() and \
           (onnx_dir / "decoder_model_merged_q4f16.onnx").exists() and \
           (onnx_dir / "decoder_model_merged_q4f16.onnx_data").exists()


def start_server():
    global _model, _initialized
    if not is_llm_available():
        logger.error("Qwen3.5-2B ONNX model not found")
        return False
    with _model_lock:
        if _model is not None:
            _model_ready.set()
            return True
        try:
            _model = Qwen3_5Onnx()
            _initialized = True
            _model_ready.set()
            logger.info("Qwen3.5-2B ONNX ready")
            return True
        except Exception as e:
            logger.error(f"Qwen3.5-2B ONNX load failed: {e}")
            _model = None
            return False


def stop_server():
    global _model
    _model_ready.clear()
    with _model_lock:
        _model = None


def is_server_running():
    return _model is not None and _model_ready.is_set()


class ContextManager:
    def __init__(self, max_pairs=10):
        self.max_pairs = max_pairs
        self.history = []

    def add(self, user_text, assistant_text):
        self.history.append((user_text, assistant_text))
        if len(self.history) > self.max_pairs:
            self.history = self.history[-self.max_pairs:]

    def clear(self):
        self.history = []


context = ContextManager()


def call_llm(prompt, system_prompt="", max_tokens=512, temperature=1.0, mtp=False,
             timeout=30, context_pairs=None):
    global _model
    if _model is None:
        if not start_server():
            raise RuntimeError("Qwen3.5-2B not available")
    if not _model_ready.wait(timeout=timeout):
        raise RuntimeError("Qwen3.5-2B not ready after timeout")

    with _inference_lock:
        try:
            sp = system_prompt or PRESETS["standard"]
            full_text = _model.apply_chat_template(sp, prompt, enable_thinking=False)
            result = _model.generate(full_text, max_new=max_tokens,
                                     temperature=temperature, top_k=20,
                                     top_p=1.0, min_p=0.0,
                                     presence_penalty=2.0,
                                     repetition_penalty=1.0)

            if not result:
                logger.info("LLM 响应为空")
                return None

            if result and context_pairs is None:
                context.add(prompt, result)

            return result if result else None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None


def load_custom_prompts():
    if CUSTOM_PROMPTS_PATH.exists():
        try:
            with open(CUSTOM_PROMPTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_custom_prompts(prompts):
    CUSTOM_PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
