"""
Test old ASR pipeline (Qwen3-ASR-0.6B ONNX CPU).
Usage:
    python test_asr.py <audio.wav> [--language zh] [--threads 6]
"""
import sys, json, time
from pathlib import Path
from onnx_infer.asr import OnnxAsrPipeline

MODEL_DIR = Path(__file__).parent / "Qwen3-ASR-0.6B-ONNX-CPU" / "onnx_models"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    language = None
    threads = 6

    args = iter(sys.argv[2:])
    for a in args:
        if a == "--language":
            language = next(args)
        elif a == "--threads":
            threads = int(next(args))

    print(f"Loading model from {MODEL_DIR}...")
    t0 = time.time()
    pipeline = OnnxAsrPipeline(onnx_dir=str(MODEL_DIR), num_threads=threads)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    print(f"Transcribing {audio_path}...")
    t0 = time.time()
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = pipeline.transcribe(audio_path, **kwargs)
    elapsed = time.time() - t0

    print(f"Result: {result['text']}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Timing: {json.dumps(result.get('timing', {}), indent=2)}")


if __name__ == "__main__":
    main()
