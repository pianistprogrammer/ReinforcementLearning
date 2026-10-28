# core/models.py — model setup and sampling utilities for MLX/MLX-LM
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx.utils import tree_flatten
from uuid import uuid4
from rl_framework.catalog import MODEL_PRESETS, DEFAULT_MODEL_ID
from rl_framework.events import emit

# Keys limited to attention projections to avoid quantized MoE layers without VJP
DEFAULT_LORA_KEYS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
}


def resolve_model_id(model_id: str) -> str:
    """Resolve a family preset, preserving explicit repository IDs and paths."""
    if not model_id.strip():
        raise ValueError("model_id must be a preset, repository ID, or local path")
    return MODEL_PRESETS.get(model_id.lower(), model_id)


def setup_model(
    model_id: str,
    *,
    lora_rank: int = 8,
    lora_layers: int = 8,
    lora_keys: set[str] | None = None,
):
    """Load an MLX-LM model, freeze base weights, and apply LoRA adapters.

    Returns (model, tokenizer) where only LoRA params are trainable.
    """
    if lora_rank < 1:
        raise ValueError("lora_rank must be positive")
    if lora_layers != -1 and lora_layers < 1:
        raise ValueError("lora_layers must be positive, or -1 for all layers")
    keys = set(DEFAULT_LORA_KEYS if lora_keys is None else lora_keys)
    if not keys:
        raise ValueError("lora_keys must contain at least one module path")

    emit("phase", phase="loading_model", message=f"Loading {resolve_model_id(model_id)}")
    model, tokenizer = load(resolve_model_id(model_id))
    layers = model.layers
    if not layers:
        raise ValueError("Model exposes no transformer layers for LoRA")
    num_layers = len(layers) if lora_layers == -1 else min(lora_layers, len(layers))
    available = {name for layer in layers[-num_layers:] for name, _ in layer.named_modules()}
    missing = keys - available
    if missing:
        raise ValueError(
            f"LoRA targets missing from the selected model layers: {sorted(missing)}. "
            "Pass lora_keys matching the model's module paths."
        )
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=num_layers,
        config={"rank": lora_rank, "scale": 20.0, "dropout": 0.0, "keys": keys},
    )
    if not any("lora_a" in name or "lora_b" in name
               for name, _ in tree_flatten(model.trainable_parameters())):
        raise RuntimeError("The selected model layers did not produce trainable LoRA adapters")
    return model, tokenizer


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def apply_chat_template(tokenizer, prompt: str, *, enable_thinking: bool | None = None) -> str:
    """Format one raw user prompt using the checkpoint's own chat template."""
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": prompt}]
        # Templates without thinking support ignore the extra variable.
        extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, **extra)
    return prompt


def encode_prompt(tokenizer, prompt: str, *, enable_thinking: bool | None = None) -> List[int]:
    """Use identical prompt IDs for sampling and scoring, with BOS added once.

    Chat templates include their own special tokens (notably Gemma's BOS).
    Base checkpoints without a chat template retain normal tokenizer encoding.
    """
    formatted = apply_chat_template(tokenizer, prompt, enable_thinking=enable_thinking)
    if _has_chat_template(tokenizer):
        ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        ids = tokenizer.encode(formatted)
    if not len(ids):
        raise ValueError("The prompt must encode to at least one token")
    return list(ids)


def sample_group(
    model,
    tokenizer,
    prompt: str,
    *,
    group_size: int = 8,
    max_tokens: int = 512,
    temperature: float = 0.8,
    enable_thinking: bool | None = None,
) -> Tuple[List[str], List[List[int]]]:
    """Return generated text and the exact sampled IDs, including a sampled EOS."""
    if group_size < 1 or max_tokens < 1:
        raise ValueError("group_size and max_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    completions, token_ids_list = [], []
    sampler = make_sampler(temp=temperature)
    prompt_ids = encode_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    for sample_index in range(group_size):
        generation_id = uuid4().hex
        emit("generation_start", generation_id=generation_id, sample_index=sample_index,
             group_size=group_size, prompt=prompt)
        segments, comp_ids = [], []
        for response in stream_generate(
            model, tokenizer, prompt=prompt_ids, max_tokens=max_tokens, sampler=sampler
        ):
            segments.append(response.text)
            comp_ids.append(response.token)
            emit("token", generation_id=generation_id, text=response.text,
                 token_id=response.token, tokens=len(comp_ids),
                 tokens_per_second=getattr(response, "generation_tps", None),
                 peak_memory_gb=getattr(response, "peak_memory", None))
        completions.append("".join(segments))
        token_ids_list.append(comp_ids)
        emit("generation_end", generation_id=generation_id, tokens=len(comp_ids))
    return completions, token_ids_list


def sequence_logprobs(model, tokenizer, prompt: str, completion_ids: List[int], *,
                      enable_thinking: bool | None = None):
    """Sum of token log-probs of a completion under model, conditioned on prompt."""
    if not len(completion_ids):
        return mx.array(0.0)
    prompt_ids = encode_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    full_ids = mx.array([prompt_ids + list(completion_ids)])
    logits = model(full_ids[:, :-1])
    logprobs = nn.log_softmax(logits, axis=-1)
    targets = full_ids[:, 1:]
    tok_lp = mx.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
    start = len(prompt_ids) - 1
    return tok_lp[:, start:].sum()
