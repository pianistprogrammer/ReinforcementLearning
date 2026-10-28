# Experimental GRPO-style sequence objective over LoRA parameters.
from __future__ import annotations
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from .models import sequence_logprobs


def collect_lora_params(model):
    flat = tree_flatten(model.trainable_parameters())
    # keep only LoRA adapter arrays
    return {k: v for k, v in flat if ("lora_a" in k or "lora_b" in k)}


def grpo_step(
    model,
    ref_model,
    tokenizer,
    optimizer,
    prompt: str,
    completions: list[str],
    token_ids_list: list[list[int]],
    rewards: list[float],
    *,
    beta: float = 0.1,
    enable_thinking: bool | None = None,
) -> float:
    rewards = np.array(rewards, dtype=np.float32)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    lora_params = collect_lora_params(model)
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found — ensure setup_model applied adapters.")

    def loss_fn(lora_params):
        # apply adapters before evaluating
        model.update(tree_unflatten(list(lora_params.items())))
        total = mx.array(0.0)
        for comp_ids, a in zip(token_ids_list, adv):
            lp = sequence_logprobs(model, tokenizer, prompt, comp_ids,
                                   enable_thinking=enable_thinking)
            ref_lp = mx.stop_gradient(sequence_logprobs(ref_model, tokenizer, prompt, comp_ids,
                                                        enable_thinking=enable_thinking))
            kl = lp - ref_lp
            total = total + (-(float(a) * lp) + beta * kl)
        return total / len(completions)

    loss, grads = mx.value_and_grad(loss_fn)(lora_params)
    optimizer.update(model, tree_unflatten(list(grads.items())))
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)
