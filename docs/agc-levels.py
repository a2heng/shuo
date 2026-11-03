#!/usr/bin/env python3
"""压缩器实验出图（对应 `cargo test --release compress_design -- --nocapture`）。

测试会导出真实（Rust `mel::compress` 处理过）的波形到 /tmp/opencode/agc/*.f32，
本脚本画三组对照：

  上：三档音量的波形，处理前/后叠在一起（看得见"全被抬到同一高度"）
  中左：输入峰值 → 输出峰值 —— 一条平线：不管多小声都对齐到 -1.8 dBFS
  中右：一句话内四段强弱 —— 22 dB 的落差被压到 5 dB 左右
  下：含 1% 爆音时，软限幅不削波、语音仍被抬满

用法：
  cd src-tauri && ~/.cargo/bin/cargo test --release compress_design -- --nocapture
  ~/venv-py3.14-cu132-torch2.14.1/bin/python docs/agc-levels.py
输出：docs/agc-levels.png
"""

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = pathlib.Path("/tmp/opencode/agc")
OUT = pathlib.Path(__file__).with_suffix(".png")

TARGET_PEAK_DB = 20 * np.log10(0.8)  # -1.94 dBFS
NO_SPEECH_DB = -62.0

plt.rcParams["font.family"] = ["Noto Serif CJK SC", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    before = np.frombuffer((SRC / f"{name}.in.f32").read_bytes(), dtype=np.float32)
    after = np.frombuffer((SRC / f"{name}.out.f32").read_bytes(), dtype=np.float32)
    return before, after


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def peak_of(x):
    return float(np.abs(x).max())


fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
fig.suptitle("压缩器（唯一环节）：逐帧低抬高降 → 整段对齐 -1.9 dBFS → 软限幅 0.98", fontsize=15)

# ---------------------------------------------------------------- 波形：两档音量
for col, (name, title) in enumerate(
    [("loud", "-3 dBFS 正常偏大"), ("mic", "-38 dBFS 本机麦克风实测（抬 +36 dB）")]
):
    before, after = load(name)
    t = np.arange(before.size) / 48000.0
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t, before, lw=0.6, color="#9e9e9e", label=f"处理前（峰值 {db(peak_of(before)):.1f} dBFS）")
    ax.plot(t, after, lw=0.6, color="#c0392b", label=f"处理后（峰值 {db(peak_of(after)):.1f} dBFS）")
    ax.set_xlim(0, 0.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("秒")
    ax.legend(fontsize=8, loc="lower right")

# ---------------------------------------------------------------- 传递曲线
ax = fig.add_subplot(gs[1, 0])
xs = np.linspace(-70, 0, 300)
points = {
    "loud": (-3.0, None), "normal": (-18.0, None), "soft": (-30.0, None),
    "mic": (-38.0, None), "quiet": (-44.0, None), "whisper": (-50.0, None),
    "silence": (-62.0, None), "noise": (-75.0, None),
}
for n in points:
    b, a = load(n)
    points[n] = (db(peak_of(b)), db(peak_of(a)))
px = [v[0] for v in points.values()]
py = [v[1] for v in points.values()]
ax.plot([-70, 0], [-70, 0], "--", color="#b0b0b0", label="1:1（不处理）")
ax.axhline(TARGET_PEAK_DB, ls=":", color="#2e86de", lw=1.2, label="目标 -1.9 dBFS")
ax.axvline(NO_SPEECH_DB, ls=":", color="#e67e22", lw=1.2, label="整段视为没说话 -62 dBFS")
ax.scatter(px, py, s=46, color="#c0392b", zorder=5, label="实测（合成波形）")
ax.set_xlim(-70, 0)
ax.set_ylim(-75, 5)
ax.set_xlabel("输入峰值 dBFS")
ax.set_ylabel("输出峰值 dBFS")
ax.set_title("输入电平 → 输出电平：非静音一律对齐 -1.9 dBFS", fontsize=11)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")

# ---------------------------------------------------------------- 段内动态
ax = fig.add_subplot(gs[1, 1])
before, after = load("compress")
per = before.size // 4
b = [db(np.sqrt((before[i * per:(i + 1) * per] ** 2).mean())) for i in range(4)]
a = [db(np.sqrt((after[i * per:(i + 1) * per] ** 2).mean())) for i in range(4)]
x = np.arange(4)
w = 0.36
ax.bar(x - w / 2, b, w, label="处理前", color="#9e9e9e")
ax.bar(x + w / 2, a, w, label="处理后", color="#c0392b")
ax.set_xticks(x, [f"第{i + 1}段" for i in range(4)])
ax.set_ylabel("RMS dBFS")
ax.set_ylim(-50, 0)
ax.grid(alpha=0.25, axis="y")
ax.set_title(f"一句话内强弱：{max(b) - min(b):.1f} dB → {max(a) - min(a):.1f} dB", fontsize=11)
ax.legend(fontsize=8)

# ---------------------------------------------------------------- 瞬态
ax = fig.add_subplot(gs[2, :])
before, after = load("transient")
t = np.arange(before.size) / 48000.0
ax.plot(t, before, lw=0.5, color="#9e9e9e", alpha=0.7, label="处理前（含 1% 爆音）")
ax.plot(t, after, lw=0.5, color="#c0392b", label="处理后（语音抬满，爆音走软限幅）")
ax.set_xlim(0, 0.06)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel("秒")
ax.set_title("瞬态：软限幅把 0.8 以上的尖峰渐进压到 0.98 以内，不硬切、不打齿", fontsize=11)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower right")

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=130)
print(f"已写出 {OUT}")
