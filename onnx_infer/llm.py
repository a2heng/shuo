import sys
import os
import json
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
        "你将获得一段语音识别文本，按以下规则处理后直接输出：\n"
        "规则：\n"
        "1. 输入可能已带有部分标点，也可能完全没有标点。仅根据语义补充缺少的必要标点（逗号、句号、问号、感叹号），已存在的标点保留不动；\n"
        "2. 不补全缺失的内容或改写用词；\n"
        "3. 标点宁少勿多，不确定时不加。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n"
        "示例：\n"
        "输入：小李明天下午的评审会议改到两点半了你的演示文稿做好了吗王总说还要加几页数据\n"
        "输出：小李，明天下午的评审会议改到两点半了，你的演示文稿做好了吗？王总说还要加几页数据。\n"
        "输入：今天路过超市买了些水果和酸奶然后去快递站取了包裹回到家才发现牛奶忘买了\n"
        "输出：今天路过超市买了些水果和酸奶，然后去快递站取了包裹，回到家才发现牛奶忘买了。\n"
        "输入：这个bug的原因找到了。数据库连接池配置不对导致超时你这边改一下配置文件然后重新部署\n"
        "输出：这个bug的原因找到了。数据库连接池配置不对导致超时，你这边改一下配置文件，然后重新部署。\n"
        "输入：下周去北京的机票订了吗酒店我看了一下有三家备选你倾向哪个\n"
        "输出：下周去北京的机票订了吗？酒店我看了一下有三家备选，你倾向哪个？\n"
        "输入：昨晚加班到九点回到家已经十点半了吃完饭又处理了几封邮件然后十二点才睡\n"
        "输出：昨晚加班到九点，回到家已经十点半了，吃完饭又处理了几封邮件，然后十二点才睡。"
    ),
    "moderate": (
        "你将获得一段语音识别文本，按以下规则处理后直接输出：\n"
        "规则：\n"
        "1. 输入可能已带有部分标点，也可能没有。根据语义补充缺少的标点，已有标点保留不动，不补全缺失的内容；\n"
        "2. 删除口癖和犹豫词（如嗯、啊、呃、那个、这个、就是说、然后、对吧、反正、大概、好像、就是、的话、之类的、这样的、对不对等）；\n"
        "3. 不替换任何实词，保持原意。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n"
        "示例：\n"
        "输入：嗯那个我觉得就是说这个方案好像不太行然后可能得改一下我再想想有没有更好的办法\n"
        "输出：我觉得这个方案不太行，可能得改一下，我再想想有没有更好的办法。\n"
        "输入：呃那个我们的服务器大概可能下周二吧要维护你提前备份一下数据\n"
        "输出：我们的服务器下周二要维护，你提前备份一下数据。\n"
        "输入：然后呢反正就是大概七八个人吧明天过来。你安排一下会议室和午饭\n"
        "输出：七八个人明天过来。你安排一下会议室和午饭。\n"
        "输入：啊对了我忘了说。昨天的会议纪要我发你了你先看一下有什么问题明天再讨论\n"
        "输出：对了，我忘了说。昨天的会议纪要我发你了，你先看一下，有什么问题明天再讨论。\n"
        "输入：这个文档我大概看了一下，然后觉得内容方面还要补充。你那边方便加一下吗\n"
        "输出：我大概看了一下，觉得内容还要补充。你那边方便加一下吗？"
    ),
    "aggressive": (
        "你将获得一段语音识别文本，按以下规则处理后直接输出：\n"
        "规则：\n"
        "1. 补充缺少的标点，已有的标点保留不动，不补全缺失的内容；\n"
        "2. 删除口癖和犹豫词（如嗯、啊、呃、那个、这个、就是说、然后、对吧、反正、大概、好像、就是、的话等）；\n"
        "3. 纠正明显因语音识别导致的同音/近音词错误，仅当确定是识别错误时才改，模棱两可时保留原文；\n"
        "4. 口语数字转为阿拉伯数字：一百二十→120、三千→3000、十五→15、一半→50%、三分之一→1/3、两点五→2.5、五点三→5.3；\n"
        "5. 特殊读法转换（电话/编号/序列场景）：幺→1、两→2、陆→6、拐→7、洞→0；\n"
        "6. 金额规范：块/毛→元/角（三块五→3.5元、五毛→0.5元）；\n"
        "7. 物理单位用符号：米/m、厘米/cm、毫米/mm、公里/km、斤/斤、克/g、公斤/kg、吨/t、毫升/ml、升/L、平方米/m²、立方米/m³、摄氏度/℃、瓦/W、千瓦/kW、伏/V、安/A、赫兹/Hz、千米每小时/km/h、米每秒/m/s、毫秒/ms、秒/s、分/min、小时/h；\n"
        "8. 数学表达式转规范：x的平方→x²、根号→√、F X→f(x)、a等于b加c→a=b+c；\n"
        "9. 连续数字合并（电话/编号等）：幺三九零零八六→1390086；\n"
        "10. 日期时间标准化：二零二六年→2026年、下午三点半→15:30。\n"
        "要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n"
        "示例：\n"
        "输入：嗯那个我觉得这个函数F X等于x的平方加一，它的导数是二x帮我查一下幺三九零零八六拐五二一\n"
        "输出：我觉得f(x)=x²+1，它的导数是2x。帮我查一下13900867521。\n"
        "输入：温度二十五度三开两个小时空调功率两千万这个大概一百二十斤重花了三块五\n"
        "输出：温度25.3℃，开2小时空调，功率2000W。这个大概120斤重，花了3.5元。\n"
        "输入：服务器下周二下午三点维护物体初始速度两点五米每秒加速度是一米每二次方秒\n"
        "输出：服务器下周二15:00维护。物体初始速度2.5m/s，加速度1m/s²。\n"
        "输入：用C加加写个类继承自BASE类然后实现接口。这个类的构造函数接收两个参数\n"
        "输出：用C++写个类继承自BASE类，实现接口。这个类的构造函数接收两个参数。\n"
        "输入：房间面积大概是二十五平方米功率是一千五百瓦电压二百二十伏\n"
        "输出：房间面积大概是25m²，功率是1500W，电压220V。"
    ),
    "aggressive_no_punc": (
        "你将获得一段语音识别文本，按以下规则处理后直接输出：\n"
        "规则：\n"
        "1. 删除口癖和犹豫词（如嗯、啊、呃、那个、这个、就是说、然后、对吧、反正、大概等）；\n"
        "2. 纠正明显因语音识别导致的同音词错误，不确定时不改；\n"
        "3. 口语数字转为阿拉伯数字：一百二十→120、三千→3000、十五→15、一半→50%、两点五→2.5；\n"
        "4. 特殊读法转换：幺→1、两→2、陆→6、拐→7、洞→0；\n"
        "5. 金额规范：块/毛→元/角（三块五→3.5元）；\n"
        "6. 物理单位用符号：米/m、厘米/cm、公斤/kg、克/g、吨/t、毫升/ml、升/L、摄氏度/℃、瓦/W、千瓦/kW、千米每小时/km/h、秒/s、分/min、小时/h；\n"
        "7. 数学表达式转规范：x的平方→x²、根号→√；\n"
        "8. 连续数字合并（电话/编号）：幺三九零零八六→1390086；\n"
        "9. 日期时间标准化：二零二六年→2026年、下午三点半→15:30；\n"
        "10. 去掉所有标点符号（句号、逗号、问号、感叹号、冒号、分号、引号等），但保留数字中的小数点。\n"
        "要求：只输出最终文本，不要加引号，不要任何解释或问候。\n"
        "示例：\n"
        "输入：嗯那个我的电话是幺三九零零八六拐五二一大概下午三点半到。你到了给我电话\n"
        "输出：我的电话是13900867521大概15:30到你到了给我电话\n"
        "输入：这个大概一百二十斤重花了三块五服务器的功率是两千万温度二十五度三\n"
        "输出：这个大概120斤重花了3.5元服务器功率是2000W温度25.3度\n"
        "输入：物体初始速度两点五米每秒。加速度一米每二次方秒帮我查一下这个电话幺三九零零八六拐五二一\n"
        "输出：物体初始速度2.5m/s加速度1m/s²帮我查一下这个电话13900867521\n"
        "输入：房间二十五平方米功率一千五百瓦电压二百二十伏\n"
        "输出：房间25平方米功率1500W电压220V"
    ),
    "translate": (
        "你将获得一段语音识别文本，按以下规则处理后直接输出：\n"
        "规则：\n"
        "1. 将文本翻译成英文，保持原意和语气；\n"
        "2. 如果原文已经是英文，则原样输出；\n"
        "3. 文本中的中文数字先转为阿拉伯数字再翻译（如三点→3:00、一百二十→120）。\n"
        "要求：只输出翻译后的文本，不要加引号，不要任何解释或问候。\n"
        "示例：\n"
        "输入：早上好你今天怎么样。下午三点有个会议大概一个小时你参加吗\n"
        "输出：Good morning, how are you today? I have a meeting at 3 PM, about one hour. Will you join?\n"
        "输入：这个项目下周五截止。请把文件发到我的邮箱然后抄送给李经理\n"
        "输出：This project is due next Friday. Please send the file to my email and cc Manager Li.\n"
        "输入：帮我订一张明天上午十点到上海的火车票。我的电话是幺三九零零八六拐五二一\n"
        "输出：Please book a train ticket to Shanghai for tomorrow at 10 AM. My phone number is 13900867521.\n"
        "输入：房间温度二十五度空调功率两千瓦开两个小时\n"
        "输出：Room temperature is 25 degrees, air conditioner power is 2000W, run for 2 hours. "
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
        self._cached_sp = None

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

    def _cache_prefix(self, system_prompt: str):
        """Cache token ids and embeddings for the fixed prompt prefix/suffix."""
        import numpy as np
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._pre_ids = self.tokenizer.encode(prefix).ids
        self._suf_ids = self.tokenizer.encode(suffix).ids
        self._pre_emb = self.embed.run(None, {"input_ids": np.array([self._pre_ids], dtype=np.int64)})[0]
        self._suf_emb = self.embed.run(None, {"input_ids": np.array([self._suf_ids], dtype=np.int64)})[0]
        self._cached_sp = system_prompt

    def generate(self, system_prompt: str, user_text: str, max_new: int = 512, temperature: float = 0.3):
        import numpy as np
        if self._cached_sp != system_prompt:
            self._cache_prefix(system_prompt)

        user_ids = self.tokenizer.encode(user_text).ids
        tokens = self._pre_ids + user_ids + self._suf_ids
        seq_len = len(tokens)

        user_emb = self.embed.run(None, {"input_ids": np.array([user_ids], dtype=np.int64)})[0]
        emb = np.concatenate([self._pre_emb, user_emb, self._suf_emb], axis=1)

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

        out = self.tokenizer.decode(ids, skip_special_tokens=True)
        return out


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


def call_llm(prompt, system_prompt="", max_tokens=512, temperature=0.3, mtp=False,
             timeout=30, context_pairs=None):
    global _model
    if _model is None:
        if not start_server():
            raise RuntimeError("Qwen3.5-2B not available")
    if not _model_ready.wait(timeout=timeout):
        raise RuntimeError("Qwen3.5-2B not ready after timeout")

    with _inference_lock:
        try:
            sp = system_prompt or "You are a text post-processing assistant. Output only the processed text without any explanation."
            result = _model.generate(sp, prompt, max_new=max_tokens, temperature=temperature)

            if not result:
                logger.info("LLM 响应为空")
                return None

            if result and "我是" in result:
                lines = result.split('\n')
                clean_lines = [l for l in lines if "我是" not in l]
                result = "\n".join(clean_lines).strip()

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
