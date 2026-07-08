# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 a2heng
#
# AI audio denoising using ONNX Runtime.
# Based on NFT_960 architecture (Stream_NFT_960).

import os, wave, struct
import numpy as np
import onnxruntime as ort
from typing import Optional

N_FFT = 960
HOP_LENGTH = 480
WIN_LENGTH = 960
SAMPLE_RATE = 48000


class AudioDenoiser:
    """Streaming audio denoiser using ONNX model (NFT_960 architecture)."""

    def __init__(self, model_path: str):
        self.session = self._load_session(model_path)
        self._reset_states()

    def _load_session(self, model_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)

    def _reset_states(self):
        self.conv_cache = np.zeros((1, 8808), dtype=np.float32)
        self.tfa_cache = np.zeros((1, 610), dtype=np.float32)
        self.inter_cache = np.zeros((1, 4608), dtype=np.float32)

    def _stft(self, x, window):
        num_frames = 1 + (len(x) - WIN_LENGTH) // HOP_LENGTH
        specs = []
        for i in range(num_frames):
            start = i * HOP_LENGTH
            frame = x[start : start + WIN_LENGTH] * window
            specs.append(np.fft.rfft(frame, N_FFT))
        return np.array(specs).T

    def _istft(self, spec, window, length):
        n_frames = spec.shape[1]
        output = np.zeros(length, dtype=np.float32)
        window_sum = np.zeros(length, dtype=np.float32)
        for i in range(n_frames):
            frame = np.fft.irfft(spec[:, i], N_FFT)[:WIN_LENGTH] * window
            start = i * HOP_LENGTH
            end = min(start + WIN_LENGTH, length)
            output[start:end] += frame[: end - start]
            window_sum[start:end] += window[: end - start] ** 2
        return output / np.maximum(window_sum, 1e-10)

    def denoise(self, audio_48k: np.ndarray) -> np.ndarray:
        """Denoise a 48kHz mono audio signal."""
        if audio_48k.ndim > 1:
            audio_48k = np.mean(audio_48k, axis=1)
        length = len(audio_48k)
        if length < WIN_LENGTH:
            return audio_48k
        self._reset_states()
        window = np.sqrt(np.hanning(WIN_LENGTH)).astype(np.float32)
        spec = self._stft(audio_48k, window)
        n_freq, n_frames = spec.shape
        enhanced = []
        for i in range(n_frames):
            spec_frame = spec[:, i]
            spec_input = np.stack([spec_frame.real, spec_frame.imag], axis=-1).astype(np.float32)
            spec_input = spec_input[np.newaxis, :, np.newaxis, :]
            outputs = self.session.run(
                None,
                {
                    "spec": spec_input,
                    "conv_cache": self.conv_cache,
                    "tfa_cache": self.tfa_cache,
                    "inter_cache": self.inter_cache,
                },
            )
            es = outputs[0][0, :, 0, 0] + 1j * outputs[0][0, :, 0, 1]
            enhanced.append(es)
            self.conv_cache = outputs[1]
            self.tfa_cache = outputs[2]
            self.inter_cache = outputs[3]
        enhanced_spec = np.array(enhanced).T
        return self._istft(enhanced_spec, window, length)[:length]


def normalize_volume(audio: np.ndarray, target_rms: float = 0.15) -> np.ndarray:
    """Normalize audio RMS to target level. Linear gain with hard clip."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    return np.clip(audio * (target_rms / rms), -1.0, 1.0)


def save_debug_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Save float32 array as 16-bit WAV file."""
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def resample_16k_to_48k(audio_16k: np.ndarray) -> np.ndarray:
    """Resample 16kHz → 48kHz (linear interpolation)."""
    n = len(audio_16k)
    out_len = n * 3
    x = np.arange(n)
    x_out = np.linspace(0, n - 1, out_len)
    return np.interp(x_out, x, audio_16k).astype(np.float32)


def resample_48k_to_16k(audio_48k: np.ndarray) -> np.ndarray:
    """Resample 48kHz → 16kHz (linear interpolation)."""
    n = len(audio_48k)
    out_len = n // 3
    x = np.arange(n)
    x_out = np.linspace(0, n - 1, out_len)
    return np.interp(x_out, x, audio_48k).astype(np.float32)
