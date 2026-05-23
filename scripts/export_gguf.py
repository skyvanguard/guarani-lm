"""Export an already-merged HF safetensors model to GGUF Q4_K_M for Ollama.

Uses llama.cpp's convert_hf_to_gguf.py + llama-quantize directly. Avoids
Unsloth's save_pretrained_gguf (which requires libcurl4-openssl-dev because
its llama.cpp build defaults to LLAMA_CURL=ON).

Usage (inside the container)::

    python3 scripts/export_gguf.py \
        --model checkpoints/cpt_v3_refine/final \
        --output checkpoints/cpt_v3_refine/gguf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model dir (merged fp16/bf16 safetensors)")
    parser.add_argument("--output", required=True, help="Output dir for GGUF files")
    parser.add_argument("--llama-cpp-dir", default="/tmp/llama.cpp")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--keep-bf16", action="store_true", help="Keep the intermediate bf16 GGUF")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    out_dir = Path(args.output).resolve()
    llama_dir = Path(args.llama_cpp_dir).resolve()

    if not model_dir.exists():
        sys.exit(f"Model dir not found: {model_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clone llama.cpp if missing
    if not llama_dir.exists():
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(llama_dir)])
    else:
        print(f"Reusing existing llama.cpp at {llama_dir}")

    # 2. Python deps for convert_hf_to_gguf.py
    req_file = llama_dir / "requirements.txt"
    if req_file.exists():
        run(["pip", "install", "--quiet", "-r", str(req_file)])

    # 3. Build llama-quantize (CURL off — we don't need server bits)
    build_dir = llama_dir / "build"
    quantize_bin = build_dir / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        run([
            "cmake", "-B", str(build_dir),
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DGGML_NATIVE=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ], cwd=llama_dir)
        run([
            "cmake", "--build", str(build_dir),
            "--target", "llama-quantize",
            "--config", "Release",
            "--parallel", "4",
        ], cwd=llama_dir)
        if not quantize_bin.exists():
            sys.exit(f"llama-quantize build failed (not at {quantize_bin})")

    # 4. Convert HF safetensors -> GGUF bf16
    bf16_path = out_dir / "model-bf16.gguf"
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {convert_script}")
    run([
        "python3", str(convert_script),
        str(model_dir),
        "--outfile", str(bf16_path),
        "--outtype", "bf16",
    ])

    # 5. Quantize bf16 -> Q4_K_M
    quant_path = out_dir / f"model-{args.quant.lower()}.gguf"
    run([str(quantize_bin), str(bf16_path), str(quant_path), args.quant])

    if not args.keep_bf16 and bf16_path.exists():
        bf16_size_gb = bf16_path.stat().st_size / 1e9
        print(f"\nRemoving intermediate bf16 GGUF ({bf16_size_gb:.2f} GB) — pass --keep-bf16 to retain it")
        bf16_path.unlink()

    print("\n=== GGUF export complete ===")
    print(f"Output dir: {out_dir}")
    for f in sorted(out_dir.glob("*.gguf")):
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
