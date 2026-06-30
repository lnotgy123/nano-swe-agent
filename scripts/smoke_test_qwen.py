from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import GenerationConfig, QwenLLMClient


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local Qwen inference.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "qwen_local.yaml"),
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function that returns the nth Fibonacci number, and explain it briefly.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))
    model_cfg = cfg["model"]

    client = QwenLLMClient(
        model_path=model_cfg["path"],
        generation_config=gen_cfg,
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
    )
    messages = [
        {
            "role": "system",
            "content": "You are a precise software engineering assistant.",
        },
        {"role": "user", "content": args.prompt},
    ]
    print(client.generate(messages))


if __name__ == "__main__":
    main()
