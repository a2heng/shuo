import sys, os, tempfile, wave, shutil, json, logging, threading, time
from pathlib import Path
from datetime import datetime
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QHBoxLayout, QDialog, QCheckBox,
    QSystemTrayIcon, QMenu, QScrollArea, QSizePolicy, QGridLayout, QStyle,
    QStyleFactory, QProxyStyle, QFrame, QTextEdit, QInputDialog, QMessageBox)
from PySide6.QtCore import (QThread, Signal, QTimer, Qt, QSize, QRect, QRectF,
    QPointF, QObject, QAbstractNativeEventFilter)
from PySide6.QtGui import QGuiApplication, QColor, QPainter, QFont, QPalette, QAction, QPen, QFontMetrics
from pynput import mouse, keyboard as kb
import pyaudio
import ctypes
import qtawesome as qta

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

if ctypes.sizeof(ctypes.c_void_p) == 8:
    _ULONG_PTR = ctypes.c_uint64
else:
    _ULONG_PTR = ctypes.c_uint32

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', ctypes.c_ushort),
        ('wScan', ctypes.c_ushort),
        ('dwFlags', ctypes.c_ulong),
        ('time', ctypes.c_ulong),
        ('dwExtraInfo', _ULONG_PTR),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ('mi', ctypes.c_byte * 32),
        ('ki', _KEYBDINPUT),
        ('hi', ctypes.c_byte * 8),
    ]

class _INPUT(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_ulong),
        ('union', _INPUT_UNION),
    ]

_SendInput = ctypes.windll.user32.SendInput
_SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
_SendInput.restype = ctypes.c_uint

def send_unicode_text(text):
    if not text:
        return
    events = []
    for ch in text:
        code = ord(ch)
        if code < 0x10000:
            codes = [code]
        else:
            code -= 0x10000
            codes = [0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)]
        for cp in codes:
            events.append(_INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=_KEYBDINPUT(0, cp, KEYEVENTF_UNICODE, 0, 0))))
            events.append(_INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=_KEYBDINPUT(0, cp, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0))))
    InputArray = _INPUT * len(events)
    _SendInput(len(events), InputArray(*events), ctypes.sizeof(_INPUT))

import i18n
import hotkey as gh
import onnx_infer.llm as llm



# ── 主题 ──────────────────────────────────────────────────────────────

class _NoAnimStyle(QProxyStyle):
    """禁用 ComboBox 弹出动画。"""
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.StyleHint.SH_ComboBox_Popup:
            return 0
        return super().styleHint(hint, option, widget, returnData)


class Theme:
    """显式颜色主题。不依赖 QPalette 传播（在 Win11 下不可靠）。"""

    # 亮色
    LIGHT = {
        "bg":         "#ffffff",
        "surface":    "#f3f3f3",
        "border":     "#e0e0e0",
        "text":       "#1e1e1e",
        "sub_text":   "#666666",
        "overlay_bg": "rgba(0,0,0,30)",
    }
    # 暗色
    DARK = {
        "bg":         "#1e1e1e",
        "surface":    "#2d2d2d",
        "border":     "#3d3d3d",
        "text":       "#f0f0f0",
        "sub_text":   "#999999",
        "overlay_bg": "rgba(0,0,0,140)",
    }

    @classmethod
    def is_dark(cls):
        return QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark

    @classmethod
    def current(cls):
        return cls.DARK if cls.is_dark() else cls.LIGHT

    @classmethod
    def accent(cls):
        """系统强调色（来自 QPalette，它对此角色是可靠的）。"""
        pal = QApplication.palette()
        try:
            return pal.color(QPalette.ColorRole.Accent)
        except AttributeError:
            return pal.color(QPalette.ColorRole.Highlight)


class _ThemeWatcher(QObject, QAbstractNativeEventFilter):
    """Win32：监听 WM_SETTINGCHANGE → 注册表变了 → 通知 MainWindow 刷新。"""
    changed = Signal()

    def nativeEventFilter(self, eventType, message):
        if eventType in (b"windows_generic_MSG", b"MSG"):
            msg = int(message)
            if msg in (0x001A, 0x031E):
                # 延迟 100ms 等注册表落定
                QTimer.singleShot(100, self.changed.emit)
        return False, 0


def _apply_title_bar(window, dark):
    try:
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(window.winId()),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value))
    except:
        pass


# ── 用户目录配置 ──────────────────────────────────────────────────────
USER_DIR = Path.home() / ".shuo"
USER_DIR.mkdir(exist_ok=True)
CONFIG_PATH = USER_DIR / "config.json"
HISTORY_PATH = USER_DIR / "history.json"
LOG_PATH = USER_DIR / "shuo.log"

# 日志配置（UTF-8）
file_handler = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("shuo")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

model_root = Path(__file__).parent / "Qwen3-ASR-0.6B-ONNX-CPU"
from onnx_infer.asr import OnnxAsrPipeline
from onnx_infer.denoise import AudioDenoiser, resample_16k_to_48k, resample_48k_to_16k

DEBUG_AUDIO = False  # 调试音频开关，True/False：启用/禁用
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEBOUNCE_MS = 500

ASR_LANGUAGES = [
    ("auto", "asr_lang.auto"),
    ("zh", "asr_lang.zh"),
    ("yue", "asr_lang.yue"),
    ("en", "asr_lang.en"),
    ("ja", "asr_lang.ja"),
    ("de", "asr_lang.de"),
    ("ko", "asr_lang.ko"),
    ("ru", "asr_lang.ru"),
    ("fr", "asr_lang.fr"),
    ("pt", "asr_lang.pt"),
    ("ar", "asr_lang.ar"),
    ("it", "asr_lang.it"),
    ("es", "asr_lang.es"),
    ("hi", "asr_lang.hi"),
    ("id", "asr_lang.id"),
    ("th", "asr_lang.th"),
    ("tr", "asr_lang.tr"),
    ("uk", "asr_lang.uk"),
    ("vi", "asr_lang.vi"),
    ("cs", "asr_lang.cs"),
    ("da", "asr_lang.da"),
    ("fil", "asr_lang.fil"),
    ("fi", "asr_lang.fi"),
    ("is", "asr_lang.is"),
    ("ms", "asr_lang.ms"),
    ("no", "asr_lang.no"),
    ("pl", "asr_lang.pl"),
    ("sv", "asr_lang.sv"),
]

DEFAULT_CONFIG = {
    "hotkey": "f2",
    "language": "en",
    "asr_lang": "auto",
    "denoise": False,
    "auto_type": True,
    "remove_punc": False,
    "save_history": False,
    "mic_device": None,
    "llm_enabled": False,
    "llm_mtp": True,
    "llm_system_prompt": "请优化以下文本：1. 修正标点符号；2. 去除口语中的口癖和犹豫词（如嗯、啊、呃、那个、就是说、然后、对吧、你知道吗、反正、就是、大概、好像等）；3. 保持原意和原始表达方式。只输出修正后的文本。",
}


class Config:
    """配置管理"""

    @staticmethod
    def load():
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    # 合并默认值
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in cfg:
                            cfg[k] = v
                    return cfg
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
        return DEFAULT_CONFIG.copy()

    @staticmethod
    def save(cfg):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")


class History:
    """历史记录管理"""

    @staticmethod
    def load():
        if HISTORY_PATH.exists():
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载历史失败: {e}")
        return []

    @staticmethod
    def save(items):
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存历史失败: {e}")

    @staticmethod
    def add(asr_text, llm_text=None):
        items = History.load()
        items.append({
            "asr_text": asr_text,
            "llm_text": llm_text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # 最多保留 500 条
        if len(items) > 500:
            items = items[-500:]
        History.save(items)
        return items


class Loader(QThread):
    done = Signal(object)
    error = Signal(str)

    def __init__(self, start_llm=False):
        super().__init__()
        self.start_llm = start_llm

    def run(self):
        try:
            logger.info("开始加载模型...")
            mdir = model_root / "onnx_models"
            pipeline = OnnxAsrPipeline(onnx_dir=str(mdir), num_threads=6)
            logger.info("ASR 模型加载完成")

            logger.info("加载降噪模型...")
            denoiser = AudioDenoiser(str(Path(__file__).parent / "onnx_infer" / "denoise.onnx"))
            logger.info("降噪模型加载完成")

            # Start LLM server if enabled
            if self.start_llm and llm.is_llm_available():
                if not llm.start_server():
                    logger.warning("LLM 启动失败")

            self.done.emit((pipeline, denoiser))
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.error.emit(str(e))


import collections


class Recorder(QThread):
    """预缓冲录音器：持续录音维护 0.5s 滚动缓冲，按住时截取 [预缓冲 + 按住期间] 的音频。"""
    finished = Signal(str)

    _PRE_BUF_SEC = 0.5

    def __init__(self, device_index=None):
        super().__init__()
        self._running = False
        self._pre_frames = collections.deque()
        self._active_frames = []
        self._segment_active = False
        self._device_index = device_index

    def run(self):
        try:
            self._running = True
            self._pre_frames.clear()
            self._active_frames.clear()
            self._segment_active = False
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=self._device_index)
            stream.start_stream()
            # 预缓冲容量：0.3s 的 chunk 数
            max_pre = max(1, int(SAMPLE_RATE * self._PRE_BUF_SEC / CHUNK))
            while self._running:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if self._segment_active:
                    self._active_frames.append(data)
                else:
                    self._pre_frames.append(data)
                    while len(self._pre_frames) > max_pre:
                        self._pre_frames.popleft()
            stream.stop_stream()
            stream.close()
            p.terminate()
            # AGC: peak 归一化到 -1 dBFS
            frames = list(self._pre_frames) + self._active_frames
            self._pre_frames.clear()
            self._active_frames.clear()
            if not frames:
                logger.warning("录音结束但无音频数据")
                return
            raw = b"".join(frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            peak = np.percentile(np.abs(samples), 99)
            if peak > 0:
                target_peak = int(0.8 * 32768)
                gain = target_peak / peak
                samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(samples.tobytes())
            self.finished.emit(tmp.name)
        except Exception as e:
            logger.error(f"录音异常: {e}")
            import traceback
            traceback.print_exc()

    def begin_segment(self):
        """按下按键：把预缓冲帧划入活动段，开始录制。"""
        self._active_frames = list(self._pre_frames)
        self._pre_frames.clear()
        self._segment_active = True

    def end_segment(self):
        """松开按键：停止活动段录制，等待 run() 结束后处理。"""
        self._segment_active = False

    def stop(self):
        self._running = False


class InferWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, pipeline, denoiser, audio_path, asr_lang=None, denoise=False):
        super().__init__()
        self.pipeline = pipeline
        self.denoiser = denoiser
        self.audio_path = audio_path
        self.asr_lang = asr_lang
        self.denoise = denoise

    def run(self):
        try:
            lang = self.asr_lang if self.asr_lang and self.asr_lang != "auto" else None
            audio_path = self.audio_path

            # Read raw audio if denoise is on
            raw_audio = None
            if self.denoise:
                import wave
                with wave.open(audio_path, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    raw_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            if self.denoise:
                from onnx_infer.denoise import save_debug_wav, resample_16k_to_48k, resample_48k_to_16k
                debug_dir = os.path.join(os.path.dirname(__file__), "debug_audio") if DEBUG_AUDIO else None
                ts = str(int(time.time() * 1000)) if DEBUG_AUDIO else None
                if DEBUG_AUDIO:
                    os.makedirs(debug_dir, exist_ok=True)
                    save_debug_wav(os.path.join(debug_dir, f"{ts}_original.wav"), raw_audio, 16000)
                audio_48k = resample_16k_to_48k(raw_audio)
                audio_denoised = self.denoiser.denoise(audio_48k)
                if DEBUG_AUDIO:
                    save_debug_wav(os.path.join(debug_dir, f"{ts}_denoised_48k.wav"), audio_denoised, 48000)
                audio_16k_denoised = resample_48k_to_16k(audio_denoised)
                if DEBUG_AUDIO:
                    save_debug_wav(os.path.join(debug_dir, f"{ts}_denoised_16k.wav"), audio_16k_denoised, 16000)
                result = self.pipeline._transcribe_chunk(audio_16k_denoised, language=lang)
            else:
                result = self.pipeline.transcribe(audio_path, language=lang)
            self.done.emit(result["text"])
        except Exception as e:
            logger.error(f"识别失败: {e}")
            self.error.emit(str(e))


class LlmWorker(QThread):
    """LLM post-processing worker. 每个实例独立处理一段 ASR 文本，支持并发。"""
    done = Signal(str, str)   # (asr_text, llm_text)
    error = Signal(str, str)  # (asr_text, msg)

    def __init__(self, asr_text, text, system_prompt="", mtp=True):
        super().__init__()
        self.asr_text = asr_text
        self.text = text
        self.system_prompt = system_prompt
        self.mtp = mtp
        # 冻结当前上下文快照，避免并发写入冲突
        self._context_pairs = llm.context.history.copy() if llm.context.history else None

    def run(self):
        try:
            result = llm.call_llm(
                self.text,
                system_prompt=self.system_prompt,
                mtp=self.mtp,
                context_pairs=self._context_pairs,
            )
            if result:
                self.done.emit(self.asr_text, result)
            else:
                self.error.emit(self.asr_text, "LLM 返回空")
        except Exception as e:
            logger.error(f"LLM 失败: {e}")
            self.error.emit(self.asr_text, str(e))


class HotkeyDialog(QDialog):
    _capture_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(i18n.tr("settings.hotkey"))
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setFixedSize(300, 120)
        layout = QVBoxLayout(self)
        self.label = QLabel(i18n.tr("settings.press_key"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.cancel_btn = QPushButton(i18n.tr("btn.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        self._captured = None
        self._mods = set()
        self._capture_signal.connect(self._on_captured)

        def on_click(x, y, btn, pressed):
            if pressed:
                m = {mouse.Button.x1: "xbutton1", mouse.Button.x2: "xbutton2"}.get(btn)
                if m:
                    self._capture_signal.emit(m)
            return False

        def on_press(key):
            if isinstance(key, kb.Key):
                k = key.name.lower()
            else:
                try: k = key.char.lower()
                except: return True
            if k in ("ctrl_l", "ctrl_r", "ctrl"): self._mods.add("ctrl"); return True
            if k in ("shift_l", "shift_r", "shift"): self._mods.add("shift"); return True
            if k in ("alt_l", "alt_r", "alt"): self._mods.add("alt"); return True
            parts = list(self._mods) + [k]
            self._capture_signal.emit("+".join(parts))
            return False

        self._ml = mouse.Listener(on_click=on_click)
        self._kl = kb.Listener(on_press=on_press)
        self._ml.start()
        self._kl.start()

    def _on_captured(self, hotkey):
        self._captured = hotkey
        self.accept()

    def get_hotkey(self):
        return self._captured

    def closeEvent(self, e):
        self._ml.stop()
        self._kl.stop()
        super().closeEvent(e)


class AboutDialog(QDialog):
    """关于对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        t = Theme.current()
        self.setWindowTitle(i18n.tr("about.title"))
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setMinimumWidth(520)
        self.setStyleSheet(f"AboutDialog {{ background:{t['bg']}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── 内容区 ──
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(14)
        cl.setContentsMargins(32, 24, 32, 24)

        title = QLabel("说 · Shuo")
        title_font = title.font()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color:{t['text']};")
        cl.addWidget(title)

        subtitle = QLabel(i18n.tr("about.subtitle"))
        sub_font = subtitle.font()
        sub_font.setPointSize(10)
        subtitle.setFont(sub_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color:{t['sub_text']};")
        cl.addWidget(subtitle)

        cl.addSpacing(8)

        accent_hex = Theme.accent().name()
        license_html = i18n.tr("about.license").format(accent=accent_hex)
        info = QLabel(
            f'<p style="line-height:1.6; color:{t["text"]};">'
            f'{license_html}</p>'
        )
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        cl.addWidget(info)

        cl.addSpacing(4)

        deps = [
            ("qtawesome", "≥1.4.2", "MIT", i18n.tr("about.dep_qtawesome")),
            ("pynput", "≥1.8.0", "LGPL-3.0", i18n.tr("about.dep_pynput")),
            ("PyAudio", "≥0.2.14", "MIT", i18n.tr("about.dep_pyaudio")),
            ("onnxruntime", "≥1.26.0", "MIT", i18n.tr("about.dep_onnx")),
            ("numpy", "≥2.4.0", "BSD-3", i18n.tr("about.dep_numpy")),
            ("librosa", "≥0.11.0", "ISC", i18n.tr("about.dep_librosa")),
            ("tokenizers", "≥0.23.0", "Apache-2.0", i18n.tr("about.dep_tokenizers")),
        ]

        grid = QWidget()
        grid_layout = QGridLayout(grid)
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        headers = [
            i18n.tr("about.col_lib"),
            i18n.tr("about.col_ver"),
            i18n.tr("about.col_lic"),
            i18n.tr("about.col_use"),
        ]
        header_font = QFont(self.font())
        header_font.setBold(True)

        for col, h in enumerate(headers):
            lbl = QLabel(h)
            lbl.setFont(header_font)
            lbl.setContentsMargins(8, 6, 8, 6)
            lbl.setStyleSheet(f"color:{t['text']};")
            grid_layout.addWidget(lbl, 0, col)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{t['border']};")
        grid_layout.addWidget(sep, 1, 0, 1, 4)

        for row, (name, ver, lic, desc) in enumerate(deps):
            r = row + 2
            row_bg = t["surface"] if row % 2 == 1 else t["bg"]
            for col, val in enumerate([name, ver, lic, desc]):
                lbl = QLabel(val)
                lbl.setContentsMargins(8, 5, 8, 5)
                lbl.setStyleSheet(f"color:{t['text']}; background:{row_bg};")
                grid_layout.addWidget(lbl, r, col)

        cl.addWidget(grid)

        layout.addWidget(content, 1)

        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"background:{t['surface']}; border-top:1px solid {t['border']};")
        bl = QHBoxLayout(btn_bar)
        bl.setContentsMargins(32, 10, 32, 10)
        bl.addStretch()
        close_btn = QPushButton(i18n.tr("btn.cancel"))
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        bl.addWidget(close_btn)
        layout.addWidget(btn_bar, 0)


class LoadingOverlay(QWidget):
    """半透明遮罩 + 旋转弧线加载动画"""

    _ARC_SPAN = 270 * 16  # 270° 弧长（Qt 使用 1/16° 单位）
    _TIMER_MS = 16        # ~60 fps

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._opacity = 0.0  # 淡入用
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._TIMER_MS)
        self.resize(parent.size() if parent else self.size())

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        if self._opacity < 1.0:
            self._opacity = min(1.0, self._opacity + 0.05)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        t = Theme.current()

        # 半透明背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, int(140 * self._opacity) if Theme.is_dark() else int(60 * self._opacity)))

        cx = self.width() // 2
        cy = self.height() // 2
        spinner_r = 20

        # 旋转弧线（系统强调色）
        accent = Theme.accent()
        pen = QPen(accent, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        start_angle = (self._angle * 16) % (360 * 16)
        painter.drawArc(QRectF(cx - spinner_r, cy - spinner_r,
                               spinner_r * 2, spinner_r * 2),
                        start_angle, self._ARC_SPAN)

        # 加载文字
        text = i18n.tr("loading.model")
        text_font = QFont(self.font().family(), 11)
        painter.setFont(text_font)
        painter.setPen(QColor(t["sub_text"]))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText(cx - tw // 2, cy + spinner_r + 28, text)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent():
            self.resize(self.parent().size())


class TextEditDialog(QDialog):
    """Large text editor dialog for editing prompt content."""

    def __init__(self, title, label, text="", parent=None):
        super().__init__(parent)
        t = Theme.current()
        self.setWindowTitle(title)
        self.setWindowIcon(_icon("fa5s.edit"))
        self.setMinimumSize(550, 350)
        self.setStyleSheet(f"TextEditDialog {{ background:{t['bg']}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color:{t['text']}; font-weight:bold;")
        layout.addWidget(lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{ background:{t['surface']}; color:{t['text']};
                         border:1px solid {t['border']}; border-radius:4px;
                         padding:8px; font-size:13px; }}
        """)
        layout.addWidget(self.text_edit, 1)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        btn_bar.addWidget(cancel_btn)
        ok_btn = QPushButton("确定")
        ok_btn.setFixedWidth(80)
        ok_btn.clicked.connect(self.accept)
        btn_bar.addWidget(ok_btn)
        layout.addLayout(btn_bar)

    def text(self):
        return self.text_edit.toPlainText()


class PromptEditorDialog(QDialog):
    """Dialog for editing custom system prompts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        t = Theme.current()
        self.setWindowTitle("编辑自定义提示词")
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.setMinimumSize(700, 500)
        self.setStyleSheet(f"PromptEditorDialog {{ background:{t['bg']}; }}")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Prompt list
        self.list_widget = QScrollArea()
        self.list_widget.setWidgetResizable(True)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setSpacing(8)
        self.list_widget.setWidget(self.list_container)
        layout.addWidget(self.list_widget, 1)

        # Load custom prompts
        self.prompts = llm.load_custom_prompts()
        self._refresh_list()

        # Add new button and close button in same row
        btn_bar = QHBoxLayout()
        add_btn = QPushButton(_icon("fa5s.plus"), " 添加")
        add_btn.clicked.connect(self._add_prompt)
        btn_bar.addWidget(add_btn)
        btn_bar.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_bar.addWidget(close_btn)
        layout.addLayout(btn_bar)

    def _refresh_list(self):
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        t = Theme.current()
        for name, text in self.prompts.items():
            row = QWidget()
            row.setStyleSheet(f"background:{t['surface']}; border-radius:6px; padding:8px;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 10, 12, 10)
            row_layout.setSpacing(12)

            # Left: name + preview (takes remaining space)
            left = QVBoxLayout()
            left.setSpacing(2)
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color:{t['text']}; font-weight:bold;")
            left.addWidget(name_label)
            preview = QLabel(text[:120] + ("…" if len(text) > 120 else ""))
            preview.setStyleSheet(f"color:{t['sub_text']}; font-size:11px;")
            preview.setWordWrap(True)
            preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            left.addWidget(preview)
            row_layout.addLayout(left, 1)

            # Right: edit + delete
            edit_btn = QPushButton("编辑")
            edit_btn.setFixedWidth(50)
            edit_btn.clicked.connect(lambda _, n=name: self._edit_prompt(n))
            row_layout.addWidget(edit_btn)
            del_btn = QPushButton("删除")
            del_btn.setFixedWidth(50)
            del_btn.clicked.connect(lambda _, n=name: self._delete_prompt(n))
            row_layout.addWidget(del_btn)

            self.list_layout.addWidget(row)

        self.list_layout.addStretch()

    def _add_prompt(self):
        dlg = TextEditDialog("添加提示词", "提示词名称:", parent=self)
        dlg.setMinimumSize(450, 250)
        # Simple name input first
        name, ok = QInputDialog.getText(self, "添加提示词", "提示词名称:")
        if not (ok and name):
            return
        # Use custom TextEditDialog for content instead of tiny QInputDialog
        dlg = TextEditDialog("添加提示词", "提示词内容:", parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        text = dlg.text()
        if text:
            self.prompts[name] = text
            llm.save_custom_prompts(self.prompts)
            self._refresh_list()

    def _edit_prompt(self, name):
        dlg = TextEditDialog("编辑提示词", f"编辑 \"{name}\" 的内容:", self.prompts.get(name, ""), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            text = dlg.text()
            self.prompts[name] = text
            llm.save_custom_prompts(self.prompts)
            self._refresh_list()

    def _delete_prompt(self, name):
        reply = QMessageBox.question(
            self, "删除提示词", f"确定删除 \"{name}\"？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self.prompts[name]
            llm.save_custom_prompts(self.prompts)
            self._refresh_list()


class ResultItem(QFrame):
    """卡片式识别结果：上方 ASR 原文，下方 LLM 优化，各带复制按钮"""

    def __init__(self, asr_text, llm_text=None, parent=None):
        super().__init__(parent)
        self._asr_text = asr_text
        self._llm_text = llm_text
        self._hover = False
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        t = Theme.current()
        accent = Theme.accent()

        # ASR row
        asr_row = QHBoxLayout()
        asr_row.setSpacing(8)
        self._asr_label = QLabel(asr_text)
        self._asr_label.setWordWrap(True)
        asr_row.addWidget(self._asr_label, 1)

        self._asr_copy = QPushButton()
        self._asr_copy.setIcon(_icon("fa5s.copy", accent.name()))
        self._asr_copy.setFixedSize(26, 26)
        self._asr_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self._asr_copy.setFlat(True)
        self._asr_copy.clicked.connect(lambda: self._copy_text(self._asr_text))
        asr_row.addWidget(self._asr_copy)
        layout.addLayout(asr_row)

        # Separator
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setFixedHeight(1)
        self._sep.setStyleSheet(f"border:none; background:{t['border']};")
        layout.addWidget(self._sep)

        # LLM row
        llm_row = QHBoxLayout()
        llm_row.setSpacing(8)
        llm_display = llm_text if llm_text else ""
        self._llm_label = QLabel(llm_display)
        self._llm_label.setWordWrap(True)
        llm_color = accent.name() if llm_text else t["sub_text"]
        self._llm_label.setStyleSheet(f"color:{llm_color};")
        llm_row.addWidget(self._llm_label, 1)

        self._llm_copy = QPushButton()
        self._llm_copy.setIcon(_icon("fa5s.copy", accent.name()))
        self._llm_copy.setFixedSize(26, 26)
        self._llm_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self._llm_copy.setFlat(True)
        self._llm_copy.clicked.connect(lambda: self._copy_text(llm_text or ""))
        llm_row.addWidget(self._llm_copy)
        layout.addLayout(llm_row)

        self._update_style()

    def _copy_text(self, text):
        QGuiApplication.clipboard().setText(text)

    def _update_style(self):
        t = Theme.current()
        accent = Theme.accent()
        bg = QColor(t["surface"]).lighter(112).name() if self._hover else t["surface"]
        border = accent.name() if self._hover else t["border"]
        self.setStyleSheet(f"""
            ResultItem {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            ResultItem QPushButton {{
                border: 1px solid {t["border"]};
                background: {t["surface"]};
                padding: 2px;
                border-radius: 6px;
            }}
            ResultItem QPushButton:hover {{
                background: {accent.lighter(160).name()};
                border-color: {accent.name()};
            }}
        """)

    def enterEvent(self, event):
        self._hover = True
        self._update_style()

    def leaveEvent(self, event):
        self._hover = False
        self._update_style()


def _icon(name, color=None):
    """创建图标。默认用主题文字色，可传 color 覆盖（如 '#ffffff'）。"""
    return qta.icon(name, color=color or Theme.current()["text"])


def _fix_device_name(name: str) -> str:
    """修复中文 Windows 上 PortAudio 返回的设备名编码乱码。"""
    if not name or '\ufffd' not in name:
        return name
    try:
        raw = name.encode('utf-8', errors='surrogateescape')
        for enc in ('gbk', 'gb2312', 'gb18030'):
            try:
                fixed = raw.decode(enc)
                if any('\u4e00' <= c <= '\u9fff' for c in fixed):
                    return fixed
            except (UnicodeDecodeError, ValueError):
                continue
    except (UnicodeEncodeError, ValueError):
        pass
    return name


def _list_input_devices():
    """枚举 MME 音频输入设备，返回去重后的设备名列表。"""
    names = []
    seen = set()
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] <= 0:
                continue
            api_info = p.get_host_api_info_by_index(info['hostApi'])
            if api_info['type'] != 2:  # MME only
                continue
            name = _fix_device_name(info['name'].strip())
            if name not in seen:
                seen.add(name)
                names.append(name)
    except Exception as e:
        logger.error(f"枚举音频设备失败: {e}")
    finally:
        p.terminate()
    return names


def _get_device_id_by_name(name: str):
    """按设备名查找 MME device index（前缀匹配）。"""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] <= 0:
                continue
            api_info = p.get_host_api_info_by_index(info['hostApi'])
            if api_info['type'] != 2:
                continue
            if _fix_device_name(info['name'].strip()).startswith(name):
                return i
    except Exception as e:
        logger.error(f"查找设备失败: {e}")
    finally:
        p.terminate()
    return None


class MainWindow(QMainWindow):
    hotkey_pressed = Signal()
    hotkey_released = Signal()

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.recorder = None
        self.worker = None
        self._temp_wav = None
        self._pending_wavs = []
        self._llm_worker = None
        self._pending_llm = []
        self.config = Config.load()
        i18n.load(self.config.get("language", "en"))
        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.on_debounce_end)
        self._rec_timer = QTimer()
        self._rec_timer.setInterval(100)
        self._rec_timer.timeout.connect(self._update_rec_time)
        self._rec_start = 0.0
        self._asr_start = 0.0
        self._llm_start = 0.0
        self.hotkey_pressed.connect(self.on_press)
        self.hotkey_released.connect(self.on_release)

        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(700, 500)
        self.setWindowIcon(_icon("fa5s.microphone"))
        # 居中显示
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move((geo.width() - self.width()) // 2,
                      (geo.height() - self.height()) // 2)

        cw = QWidget()
        cw.setObjectName("centralWidget")
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 内容包装器（工具栏+结果，带边距）
        content_wrap = QWidget()
        content_wrap.setObjectName("contentWrap")
        cw_layout = QVBoxLayout(content_wrap)
        cw_layout.setContentsMargins(12, 12, 12, 0)
        cw_layout.setSpacing(8)

        # 状态栏组件（提前创建，供 _build_prompt_menu 引用）
        def _sb_section(w, label, val0, wd=20):
            """固定宽度的状态栏段：标题左对齐 + 文字右对齐"""
            w.setFixedWidth(wd)
            lay = QHBoxLayout(w)
            lay.setContentsMargins(1, 0, 1, 0)
            lay.setSpacing(3)
            lbl = QLabel(label)
            val = QLabel(val0)
            for q in (lbl, val):
                q.setStyleSheet("font-size:11px;")
            lay.addWidget(lbl)
            lay.addStretch()
            lay.addWidget(val)
            return lbl, val

        self._sb = QFrame()
        self._sb.setObjectName("sb")
        self_llm_w = QWidget();     self._sb_llm_l, self._sb_llm_v = _sb_section(self_llm_w, "LLM", "✗", 50)
        # 提示词：居中
        self_pr_w = QWidget();      self._sb_pr_v = QLabel("标准 - 只修标点")
        self._sb_pr_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sb_pr_v.setStyleSheet("font-size:11px;")
        _pr_lay = QHBoxLayout(self_pr_w)
        _pr_lay.setContentsMargins(0,0,0,0)
        _pr_lay.addWidget(self._sb_pr_v)
        self_pr_w.setFixedWidth(160)
        self._sb_pr_l = QLabel()  # dummy
        self_rec_w = QWidget();     self._sb_rec_l, self._sb_rec_v = _sb_section(self_rec_w, "录音", "--", 75)
        self_asr_w = QWidget();     self._sb_asr_l, self._sb_asr_v = _sb_section(self_asr_w, "ASR", "--", 70)
        self_llm_t_w = QWidget();   self._sb_llm_t_l, self._sb_llm_t_v = _sb_section(self_llm_t_w, "优化", "--", 70)

        sb_lay = QHBoxLayout(self._sb)
        sb_lay.setContentsMargins(12, 0, 12, 0)
        sb_lay.setSpacing(0)
        self._sb_pips = []
        def _pip():
            l = QLabel(" │ ")
            l.setStyleSheet("font-size:11px;")
            self._sb_pips.append(l)
            return l
        sb_lay.addWidget(self_llm_w)
        sb_lay.addWidget(_pip())
        sb_lay.addWidget(self_pr_w)
        sb_lay.addWidget(_pip())
        sb_lay.addWidget(self_rec_w)
        sb_lay.addWidget(_pip())
        sb_lay.addWidget(self_asr_w)
        sb_lay.addWidget(_pip())
        sb_lay.addWidget(self_llm_t_w)
        sb_lay.addStretch()

        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # 左侧：设置菜单 + 快捷键提示
        self.settings_btn = QPushButton(_icon("fa5s.cog"), f"  {i18n.tr('btn.settings')}")
        self.settings_btn.setFixedHeight(32)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_settings_menu()
        toolbar.addWidget(self.settings_btn)

        self.hotkey_label = QLabel()
        self.hotkey_label.setStyleSheet("font-size:11px;")
        toolbar.addWidget(self.hotkey_label)

        toolbar.addStretch()

        # 右侧：ASR 识别语言 + 麦克风
        self.asr_label = QLabel(i18n.tr("settings.asr_lang_label"))
        self.asr_label.setStyleSheet("font-size:11px;")
        toolbar.addWidget(self.asr_label)
        self.asr_lang_box = QComboBox()
        self.asr_lang_box.setFixedWidth(140)
        self.asr_lang_box.setMaxVisibleItems(20)
        self.asr_lang_box.setToolTip(i18n.tr("settings.asr_lang_tip"))
        self._populate_asr_lang()
        current_asr = self.config.get("asr_lang", "auto")
        for i, (name, _) in enumerate(ASR_LANGUAGES):
            if name == current_asr:
                self.asr_lang_box.setCurrentIndex(i)
                break
        self.asr_lang_box.currentIndexChanged.connect(self._on_asr_lang_changed)
        toolbar.addWidget(self.asr_lang_box)

        self.mic_label = QLabel(i18n.tr("settings.mic"))
        self.mic_label.setStyleSheet("font-size:11px;")
        toolbar.addWidget(self.mic_label)
        self.mic_box = QComboBox()
        self.mic_box.setFixedWidth(200)
        self.mic_box.setMaxVisibleItems(20)
        self._populate_mic_list()
        self.mic_box.currentIndexChanged.connect(self._on_mic_changed)
        toolbar.addWidget(self.mic_box)

        cw_layout.addLayout(toolbar)

        # 结果区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                border: none; background: transparent;
                width: 8px; margin: 2px;
            }
            QScrollBar::handle:vertical {
                border-radius: 4px;
                background: palette(mid);
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: palette(dark);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical { background: none; }
        """)

        self.result_container = QWidget()
        self.result_container.setObjectName("resultContainer")
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.setContentsMargins(6, 6, 6, 6)
        self.result_layout.setSpacing(10)
        self.result_layout.addStretch()

        scroll.setWidget(self.result_container)
        cw_layout.addWidget(scroll, 1)

        main_layout.addWidget(content_wrap, 1)

        # 分割线
        self._sb_top = QFrame()
        self._sb_top.setFrameShape(QFrame.Shape.HLine)
        self._sb_top.setFixedHeight(1)
        main_layout.addWidget(self._sb_top)

        main_layout.addWidget(self._sb)

        # 托盘
        self.setup_tray()

        # 加载历史记录
        self.load_history()

        # 从配置加载热键
        gh.load()

        # 初始化
        self.loader = Loader(start_llm=self.config.get("llm_enabled", False))
        self.loader.done.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_load_error)
        self.loader.start()

        self.overlay = LoadingOverlay(cw)
        self.overlay.raise_()

        self._refresh_icons()

        # 监听系统主题切换（Win32 主 + Qt 信号后备）
        self._theme_watcher = _ThemeWatcher()
        self._theme_watcher.changed.connect(self._apply_theme)
        QApplication.instance().installNativeEventFilter(self._theme_watcher)
        QApplication.styleHints().colorSchemeChanged.connect(
            lambda _: QTimer.singleShot(50, self._apply_theme))

        # 首次应用主题
        self._apply_theme()

        # 设置窗口标题栏深色/浅色
        self.winId()

    def setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(_icon("fa5s.microphone"))
        self.tray.setToolTip(i18n.tr("app.title"))
        menu = QMenu()
        self.quit_action = QAction(i18n.tr("tray.quit"), self)
        self.quit_action.triggered.connect(self.quit_app)
        menu.addAction(self.quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def quit_app(self):
        gh.stop()
        self._cleanup_temp_wav()
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(1000)
        if self.worker and self.worker.isRunning():
            self.worker.wait(1000)
        # Wait for LLM worker
        if self._llm_worker and self._llm_worker.isRunning():
            self._llm_worker.wait(2000)
        # Stop LLM server
        llm.stop_server()
        QApplication.instance().removeNativeEventFilter(self._theme_watcher)
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def _apply_theme(self):
        """读当前系统主题 → 把显式颜色写到所有需要刷新的组件。"""
        t = Theme.current()
        dark = Theme.is_dark()

        # ── 主窗口背景 ──
        cw = self.centralWidget()
        if cw:
            cw.setStyleSheet(
                f"QWidget#centralWidget {{ background: {t['bg']}; }}"
                f"QWidget#contentWrap {{ background: {t['bg']}; }}")
            cw.repaint()

        # ── 标题栏 ──
        _apply_title_bar(self, dark)

        # ── 分割线 ──
        self._sb_top.setStyleSheet(f"border:none; background:{t['border']};")

        # ── 状态栏 ──
        self._sb.setStyleSheet(f"QFrame#sb {{ background:transparent; }}")
        for attr in ('_sb_llm_l', '_sb_pr_v',
                     '_sb_rec_l', '_sb_rec_v', '_sb_asr_l', '_sb_asr_v',
                     '_sb_llm_t_l', '_sb_llm_t_v'):
            sq = getattr(self, attr)
            c = t['text']
            if attr == '_sb_llm_v':
                sq.setStyleSheet(f"font-size:11px;")  # color set by _set_llm
            else:
                sq.setStyleSheet(f"font-size:11px; color:{c};")
        for p in self._sb_pips:
            p.setStyleSheet(f"font-size:11px; color:{t['border']};")

        # ── 所有结果卡片立即重绘 ──
        for i in range(self.result_layout.count()):
            w = self.result_layout.itemAt(i).widget()
            if w:
                w.repaint()

        # ── 图标 ──
        self.setWindowIcon(_icon("fa5s.microphone"))
        self.tray.setIcon(_icon("fa5s.microphone"))
        self._refresh_icons()

    def _refresh_icons(self):
        """刷新图标颜色"""
        if not hasattr(self, 'settings_btn'):
            return
        self.settings_btn.setIcon(_icon("fa5s.cog"))
        self.settings_btn.setText(f"  {i18n.tr('btn.settings')}")

    def load_history(self):
        """加载历史记录（最新的在最上面）"""
        items = History.load()
        for item in items:
            asr_text = item.get("asr_text", "") or item.get("text", "")
            llm_text = item.get("llm_text")
            if asr_text:
                widget = ResultItem(asr_text, llm_text)
                self.result_layout.insertWidget(0, widget)

    def _build_settings_menu(self):
        """构建设置下拉菜单。"""
        menu = QMenu(self)

        # ── 复选框区 ──
        self._auto_type_action = menu.addAction(i18n.tr("settings.auto_type"))
        self._auto_type_action.setCheckable(True)
        self._auto_type_action.setChecked(self.config.get("auto_type", True))
        self._auto_type_action.toggled.connect(self.on_auto_type_changed)

        self._history_action = menu.addAction(i18n.tr("settings.history"))
        self._history_action.setCheckable(True)
        self._history_action.setChecked(self.config.get("save_history", False))
        self._history_action.toggled.connect(self.on_history_changed)

        self._punc_action = menu.addAction(i18n.tr("settings.remove_punc"))
        self._punc_action.setCheckable(True)
        self._punc_action.setChecked(self.config.get("remove_punc", False))
        self._punc_action.toggled.connect(self.on_remove_punc_changed)
        lang = self.config.get("language", "en")
        self._punc_action.setVisible(lang == "zh")

        self._denoise_action = menu.addAction(i18n.tr("settings.denoise"))
        self._denoise_action.setCheckable(True)
        self._denoise_action.setChecked(self.config.get("denoise", False))
        self._denoise_action.toggled.connect(self.on_denoise_changed)

        menu.addSeparator()

        # ── LLM 后处理区 ──
        self._llm_action = menu.addAction("LLM 后处理")
        self._llm_action.setCheckable(True)
        self._llm_action.setChecked(self.config.get("llm_enabled", False))
        self._llm_action.toggled.connect(self.on_llm_changed)
        self._llm_action.setEnabled(llm.is_llm_available())

        # System prompt submenu
        self._prompt_menu = menu.addMenu(_icon("fa5s.file-alt"), "系统提示词")
        self._prompt_menu.setToolTipsVisible(True)
        self._build_prompt_menu()

        menu.addSeparator()

        # ── 配置区 ──
        menu.addAction(_icon("fa5s.keyboard"), i18n.tr("settings.hotkey"), self.open_settings)

        # 界面语言子菜单
        self._lang_menu = menu.addMenu(_icon("fa5s.language"), i18n.tr("settings.ui_lang"))
        for code, name in i18n.list_locales():
            self._lang_menu.addAction(name, lambda c=code: self._switch_lang(c))

        menu.addSeparator()

        # ── 操作区 ──
        menu.addAction(_icon("fa5s.trash"), i18n.tr("btn.clear_history"), self.clear_history)
        menu.addAction(_icon("fa5s.info-circle"), i18n.tr("btn.about"), self.open_about)
        menu.addAction(_icon("fa5s.sign-out-alt"), i18n.tr("tray.quit"), self.quit_app)

        self.settings_btn.setMenu(menu)

    def open_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def open_settings(self):
        gh.stop()
        dlg = HotkeyDialog()
        if dlg.exec() == QDialog.DialogCode.Accepted:
            hk = dlg.get_hotkey()
            if hk:
                self.config["hotkey"] = hk
                Config.save(self.config)
                gh.save(hk)
        gh.start(on_down=self.hotkey_pressed.emit, on_up=self.hotkey_released.emit)
        self.update_hint()

    def _populate_asr_lang(self):
        self.asr_lang_box.clear()
        for name, key in ASR_LANGUAGES:
            self.asr_lang_box.addItem(i18n.tr(key), name)

    def _on_asr_lang_changed(self, idx):
        code = self.asr_lang_box.itemData(idx)
        self.config["asr_lang"] = code
        Config.save(self.config)

    def _populate_mic_list(self):
        self.mic_box.blockSignals(True)
        self.mic_box.clear()
        self.mic_box.addItem(i18n.tr("settings.mic_default"), None)
        for name in _list_input_devices():
            self.mic_box.addItem(name, name)
        saved = self.config.get("mic_device")
        if saved:
            for i in range(1, self.mic_box.count()):
                if self.mic_box.itemData(i) == saved:
                    self.mic_box.setCurrentIndex(i)
                    break
            else:
                logger.warning(f"已保存的麦克风 '{saved}' 不可用，回退到默认")
        self.mic_box.blockSignals(False)

    def _on_mic_changed(self, idx):
        dev = self.mic_box.itemData(idx)
        self.config["mic_device"] = dev
        Config.save(self.config)
        if self.pipeline:
            self._start_recorder()

    def _get_mic_device_index(self):
        """获取当前选中的麦克风 device index，None 表示系统默认。"""
        idx = self.mic_box.currentIndex()
        if idx <= 0:
            return None
        name = self.mic_box.itemData(idx)
        return _get_device_id_by_name(name) if name else None

    def update_hint(self):
        hk = gh.get()
        self.hotkey_label.setText(i18n.tr("hint.hotkey").format(key=hk.upper()))

    def on_auto_type_changed(self, checked):
        self.config["auto_type"] = checked
        Config.save(self.config)

    def on_history_changed(self, checked):
        self.config["save_history"] = checked
        Config.save(self.config)

    def on_remove_punc_changed(self, checked):
        self.config["remove_punc"] = checked
        Config.save(self.config)

    def on_denoise_changed(self, checked):
        self.config["denoise"] = checked
        Config.save(self.config)

    def on_llm_changed(self, checked):
        self.config["llm_enabled"] = checked
        Config.save(self.config)
        if checked:
            self._set_llm("⋯")
            def _start():
                if not llm.is_server_running():
                    llm.start_server()
                QTimer.singleShot(0, lambda: self._set_llm("✓"))
            threading.Thread(target=_start, daemon=True).start()
        else:
            llm.stop_server()
            self._set_llm("✗")
        self._log_bar()

    def _build_prompt_menu(self):
        """Build system prompt submenu with presets and custom options."""
        self._prompt_menu.clear()

        # Preset display names with full prompt visible
        preset_names = {
            "standard": "标准 - 只修标点",
            "moderate": "轻度 - 去口癖",
            "aggressive": "激进 - 语义重写",
            "aggressive_no_punc": "激进 - 重写去标点",
            "programmer": "程序员 - 代码术语修正",
            "translate": "翻译 - 译成英文",
            "units": "单位数字 - 格式化",
        }

        from PySide6.QtGui import QActionGroup
        self._prompt_group = QActionGroup(self)
        self._prompt_group.setExclusive(True)

        current = self.config.get("llm_system_prompt", "")
        first_preset = None

        # Presets showing clean name, full prompt in display below
        for key, text in llm.PRESETS.items():
            name = preset_names.get(key, key)
            action = self._prompt_menu.addAction(name)
            action.setToolTip(text)
            action.setCheckable(True)
            action.setChecked(current == text)
            action.triggered.connect(lambda checked=False, t=text: self._set_system_prompt(t))
            self._prompt_group.addAction(action)
            if first_preset is None:
                first_preset = action

        self._prompt_menu.addSeparator()

        # Custom prompts
        custom = llm.load_custom_prompts()
        for name, text in custom.items():
            action = self._prompt_menu.addAction(f"[自定义] {name}")
            action.setToolTip(text)
            action.setCheckable(True)
            action.setChecked(current == text)
            action.triggered.connect(lambda checked=False, t=text: self._set_system_prompt(t))
            self._prompt_group.addAction(action)

        if custom:
            self._prompt_menu.addSeparator()

        self._prompt_menu.addAction(_icon("fa5s.edit"), "管理自定义提示词…", self.manage_custom_prompts)

        # 初始化状态栏提示词
        checked = any(a.isChecked() for a in self._prompt_group.actions())
        if checked:
            self._sb_pr_v.setText(self._prompt_name(current))
            self._sb_pr_v.setToolTip(current)
        elif current:
            # 当前提示词已被修改（如编辑自定义提示词后），保持 config 不变
            self._sb_pr_v.setText(self._prompt_name(current))
            self._sb_pr_v.setToolTip(current)
        elif first_preset:
            first_preset.setChecked(True)
            first_text = list(llm.PRESETS.values())[0]
            self._set_system_prompt(first_text)

    def _set_system_prompt(self, prompt):
        self.config["llm_system_prompt"] = prompt
        Config.save(self.config)
        self._sb_pr_v.setText(self._prompt_name(prompt))
        self._sb_pr_v.setToolTip(prompt)
        self._log_bar()

    def _prompt_name(self, text):
        if not text:
            return ""
        names = {
            "standard": "标准 - 只修标点",
            "moderate": "轻度 - 去口癖",
            "aggressive": "激进 - 语义重写",
            "aggressive_no_punc": "激进 - 重写去标点",
            "programmer": "程序员 - 代码术语修正",
            "translate": "翻译 - 译成英文",
            "units": "单位数字 - 格式化",
        }
        for key, t in llm.PRESETS.items():
            if t == text:
                return names.get(key, key)
        for name, t in llm.load_custom_prompts().items():
            if t == text:
                return f"[自定义] {name}"
        # fallback: show first line of prompt text
        first = text.split("\n")[0].strip()
        return first[:40] + "…" if len(first) > 40 else first

    def manage_custom_prompts(self):
        old_prompts = llm.load_custom_prompts()
        current = self.config.get("llm_system_prompt", "")
        # find which custom prompt name was selected
        old_name = next((n for n, t in old_prompts.items() if t == current), None)

        dlg = PromptEditorDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_prompts = llm.load_custom_prompts()
            if old_name is not None and old_name in new_prompts:
                new_text = new_prompts[old_name]
                if new_text != current:
                    self.config["llm_system_prompt"] = new_text
                    Config.save(self.config)
            self._build_prompt_menu()

    def clear_history(self):
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle(i18n.tr("dialog.clear_title"))
        box.setText(i18n.tr("dialog.clear_msg"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setWindowIcon(_icon("fa5s.microphone"))
        box.setIconPixmap(_icon("fa5s.microphone").pixmap(32, 32))
        reply = box.exec()
        if reply == QMessageBox.StandardButton.Yes:
            History.save([])
            # 清空界面
            while self.result_layout.count() > 1:
                item = self.result_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _switch_lang(self, lang):
        self.config["language"] = lang
        Config.save(self.config)
        i18n.load(lang)
        self._build_settings_menu()
        self.setWindowTitle(i18n.tr("app.title"))
        self.tray.setToolTip(i18n.tr("app.title"))
        self.settings_btn.setText(f"  {i18n.tr('btn.settings')}")
        self.asr_lang_box.setToolTip(i18n.tr("settings.asr_lang_tip"))
        self.quit_action.setText(i18n.tr("tray.quit"))
        self.asr_label.setText(i18n.tr("settings.asr_lang_label"))
        self.mic_label.setText(i18n.tr("settings.mic"))
        self._populate_asr_lang()
        current_asr = self.config.get("asr_lang", "auto")
        for i in range(self.asr_lang_box.count()):
            if self.asr_lang_box.itemData(i) == current_asr:
                self.asr_lang_box.setCurrentIndex(i)
                break
        self._populate_mic_list()
        self.update_hint()

    def on_model_loaded(self, payload):
        self.pipeline, self.denoiser = payload
        self.update_hint()
        self.setFocus()
        gh.start(on_down=self.hotkey_pressed.emit, on_up=self.hotkey_released.emit)
        self.overlay.hide()
        self._start_recorder()
        if llm.is_server_running():
            self._set_llm("✓")

    def _start_recorder(self):
        """启动持续录音（预缓冲模式）。"""
        if self.recorder:
            if self.recorder.isRunning():
                self.recorder.stop()
                self.recorder.wait(2000)
            try: self.recorder.finished.disconnect()
            except Exception: pass
            self.recorder.deleteLater()
            self.recorder = None
        dev = self._get_mic_device_index()
        self.recorder = Recorder(device_index=dev)
        self.recorder.finished.connect(self.on_recorded)
        self.recorder.start()

    def on_load_error(self, msg):
        self.overlay.hide()

    def on_press(self):
        if self.debounce.isActive():
            self.debounce.stop()
            return
        if self.recorder and self.recorder.isRunning():
            self.recorder.begin_segment()
            self._rec_start = time.perf_counter()
            self._rec_timer.start()

    def on_release(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.end_segment()
            self.debounce.start(DEBOUNCE_MS)

    def on_debounce_end(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()

    def on_recorded(self, path):
        self._rec_timer.stop()
        dur = time.perf_counter() - self._rec_start
        self._sb_rec_v.setText(f"{dur:.1f}s")
        # 录音器已停止，重启持续录音
        self._start_recorder()
        # 如果正在识别，排队
        if self.worker and self.worker.isRunning():
            self._pending_wavs.append(path)
            return
        self._start_infer(path)

    def _set_llm(self, s):
        self._sb_llm_v.setText(s)
        c = {"✓": "#22c55e", "✗": "#ef4444", "⋯": "#eab308"}.get(s, "")
        if c:
            self._sb_llm_v.setStyleSheet(f"font-size:11px; color:{c};")

    def _log_bar(self):
        logger.info(f"LLM:{self._sb_llm_v.text()} | '{self._sb_pr_v.text()}' | 录音 {self._sb_rec_v.text()} | ASR {self._sb_asr_v.text()} | 优化 {self._sb_llm_t_v.text()}")

    def _update_rec_time(self):
        dur = time.perf_counter() - self._rec_start
        self._sb_rec_v.setText(f"{dur:.1f}s")

    def _start_infer(self, path):
        # 清理旧 worker
        if self.worker:
            try: self.worker.done.disconnect()
            except Exception: pass
            try: self.worker.error.disconnect()
            except Exception: pass
            self.worker.deleteLater()
        self._temp_wav = path
        self._asr_start = time.perf_counter()
        self._sb_asr_v.setText("…")
        self.worker = InferWorker(self.pipeline, self.denoiser, path,
                                   self.config.get("asr_lang", "auto"),
                                   denoise=self.config.get("denoise", False))
        self.worker.done.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_result(self, text):
        self._cleanup_temp_wav()
        dur = time.perf_counter() - self._asr_start
        self._sb_asr_v.setText(f"{dur:.1f}s")
        if text.strip():
            if self.config.get("llm_enabled", False) and llm.is_llm_available():
                self._start_llm(text)
                return
            self._log_bar()
            self._finalize_text(text, None)
        self._process_next()

    def _start_llm(self, text):
        """排队处理 LLM，与 ASR 一样串行执行。"""
        if self._llm_worker and self._llm_worker.isRunning():
            self._pending_llm.append(text)
            return
        self._do_llm(text)

    def _do_llm(self, text):
        if self._llm_worker:
            try: self._llm_worker.done.disconnect()
            except Exception: pass
            try: self._llm_worker.error.disconnect()
            except Exception: pass
            self._llm_worker.deleteLater()
        system_prompt = self.config.get("llm_system_prompt", "")
        self._llm_start = time.perf_counter()
        self._sb_llm_t_v.setText("…")
        self._llm_worker = LlmWorker(
            text, text,
            system_prompt=system_prompt,
            mtp=self.config.get("llm_mtp", True),
        )
        self._llm_worker.done.connect(self.on_llm_result)
        self._llm_worker.error.connect(self.on_llm_error)
        self._llm_worker.finished.connect(self._on_llm_finished)
        self._llm_worker.start()

    def _on_llm_finished(self):
        self._llm_worker = None
        if self._pending_llm:
            text = self._pending_llm.pop(0)
            self._do_llm(text)

    def on_llm_result(self, asr_text, llm_text):
        dur = time.perf_counter() - self._llm_start
        self._sb_llm_t_v.setText(f"{dur:.1f}s")
        self._log_bar()
        self._finalize_text(asr_text, llm_text)

    def on_llm_error(self, asr_text, msg):
        dur = time.perf_counter() - self._llm_start
        self._sb_llm_t_v.setText("✗")
        self._log_bar()
        logger.warning(f"LLM 处理失败 (asr={asr_text[:50]}): {msg}")
        self._finalize_text(asr_text, None)

    def _cleanup_llm_worker(self, worker):
        """移除旧的 LLM worker（不再使用，保留兼容）。"""
        worker.deleteLater()

    def _finalize_text(self, asr_text, llm_text):
        text = llm_text or asr_text
        if text.strip():
            lang = self.config.get("language", "en")
            if lang == "zh" and self.config.get("remove_punc", False):
                import re
                _PUNC = '，。！？、；：""\'\'（）【】《》…—,.!?;:\"\'()[]{}<>~·～—'
                text = re.sub('[' + re.escape(_PUNC) + ']', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if llm_text:
                    llm_text = re.sub('[' + re.escape(_PUNC) + ']', ' ', llm_text)
                    llm_text = re.sub(r'\s+', ' ', llm_text).strip()
            if self.config.get("save_history", False):
                History.add(asr_text, llm_text)
            while self.result_layout.count() > 101:
                item = self.result_layout.takeAt(self.result_layout.count() - 2)
                if item and item.widget():
                    item.widget().deleteLater()
            item_widget = ResultItem(asr_text, llm_text)
            self.result_layout.insertWidget(0, item_widget)
            if self.config.get("auto_type", True):
                auto_text = llm_text or asr_text
                send_unicode_text(auto_text)
        self._process_next()

    def on_error(self, msg):
        self._cleanup_temp_wav()
        self._sb_asr_v.setText("✗")
        self._log_bar()
        self._process_next()

    def _process_next(self):
        if self._pending_wavs:
            path = self._pending_wavs.pop(0)
            self._start_infer(path)

    def _cleanup_temp_wav(self):
        if self._temp_wav:
            try: Path(self._temp_wav).unlink(missing_ok=True)
            except Exception: pass
            self._temp_wav = None
        # 清理旧识别线程
        if self.worker and not self.worker.isRunning():
            self.worker.deleteLater()
            self.worker = None

    def keyPressEvent(self, e):
        pass

    def keyReleaseEvent(self, e):
        pass


if __name__ == "__main__":
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(sys.argv)
        app.setApplicationName("说 · Shuo")
        app.setStyle(_NoAnimStyle(QStyleFactory.create("windows11")))

        # 字体：只设 family + size，不放全局样式表（会破坏原生调色板）
        font = app.font()
        font.setFamilies(["Segoe UI", "Microsoft YaHei", "sans-serif"])
        font.setPointSize(9)
        app.setFont(font)

        app.setWindowIcon(_icon("fa5s.microphone"))
        i18n.load("en")  # 构造 UI 前必须加载，MainWindow 内会按用户配置重新加载
        logger.info("应用启动")
        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"应用异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
