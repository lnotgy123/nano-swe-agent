from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False


class QwenLLMClient:
    """Small wrapper around Qwen chat inference.

    This is intentionally minimal. The agent should depend on this interface
    rather than on raw transformers calls.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        generation_config: GenerationConfig | None = None,
        torch_dtype: str | torch.dtype = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.generation_config = generation_config or GenerationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        config = self.generation_config
        generate_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": config.do_sample,
        }
        generate_kwargs.update(overrides)

        if not generate_kwargs.get("do_sample", False):
            generate_kwargs.pop("temperature", None)
            generate_kwargs.pop("top_p", None)

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        return len(token_ids)
