"""Offline checks using actual MLX-LM architectures, with no model downloads."""
import math
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma, gemma2, gemma3_text, qwen2, qwen3

from rl_framework.core import models
from rl_framework.core.grpo import grpo_step


class BaseTokenizer:
    chat_template = None

    def encode(self, text):
        return [0, 1]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("base checkpoints must not use a chat template")


def make_model(module):
    config = dict(model_type=module.__name__.rsplit('.', 1)[-1], hidden_size=64,
                  num_hidden_layers=2, intermediate_size=128, num_attention_heads=2,
                  head_dim=32, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=1,
                  max_position_embeddings=128, rope_theta=10000, tie_word_embeddings=True,
                  sliding_window_pattern=2)
    return module.Model(module.ModelArgs.from_dict(config))


@pytest.mark.parametrize('module', [gemma, gemma2, gemma3_text, qwen2, qwen3])
@pytest.mark.parametrize('quantized', [False, True])
def test_native_architecture_lora_and_grpo(module, quantized):
    mx.random.seed(7)
    model, ref = make_model(module), make_model(module)
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
        nn.quantize(ref, group_size=32, bits=4)
    ref.update(model.parameters())
    ref.freeze()
    tokenizer = BaseTokenizer()
    with patch.object(models, 'load', return_value=(model, tokenizer)):
        model, _ = models.setup_model('local-checkpoint', lora_rank=2, lora_layers=1)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert len(trainable) == 8  # A/B for each attention projection in the last block.
    assert all('layers.1.' in key and ('lora_a' in key or 'lora_b' in key)
               for key in trainable)
    before = {key: mx.array(value) for key, value in tree_flatten(model.parameters())}
    optimizer = optim.Adam(learning_rate=.01)
    loss = grpo_step(model, ref, tokenizer, optimizer, 'raw', ['a', 'b'], [[2], [3]], [0., 1.])
    assert math.isfinite(loss)
    assert int(optimizer.step) == 1
    after = dict(tree_flatten(model.parameters()))
    assert any(bool(mx.any(before[key] != after[key])) for key in trainable)
    assert all(bool(mx.array_equal(value, after[key])) for key, value in before.items()
               if key not in trainable)


@pytest.mark.parametrize('name', ['gemma', 'qwen', 'gpt-oss'])
def test_presets_load_through_shared_core(name):
    with patch.object(models, 'load', return_value=(make_model(qwen3), BaseTokenizer())) as load:
        models.setup_model(name, lora_rank=2, lora_layers=-1)
    load.assert_called_once_with(models.MODEL_PRESETS[name])
    assert models.resolve_model_id(name.upper()) == models.MODEL_PRESETS[name]
    assert models.resolve_model_id('/tmp/my-model') == '/tmp/my-model'
    assert models.resolve_model_id('mlx-community/another-model') == 'mlx-community/another-model'


def test_invalid_adapter_targets_fail_before_training():
    with patch.object(models, 'load', return_value=(make_model(gemma), BaseTokenizer())):
        with pytest.raises(ValueError, match='LoRA targets missing'):
            models.setup_model('gemma', lora_keys={'self_attn.missing'})


@pytest.mark.parametrize('options', [{'lora_layers': 0}, {'lora_layers': -2},
                                     {'lora_rank': 0}, {'lora_keys': set()}])
def test_invalid_adapter_options_fail_before_loading(options):
    with patch.object(models, 'load') as load:
        with pytest.raises(ValueError):
            models.setup_model('qwen', **options)
    load.assert_not_called()


def test_custom_targets_and_layer_count_clamping():
    with patch.object(models, 'load', return_value=(make_model(qwen2), BaseTokenizer())):
        model, _ = models.setup_model('qwen', lora_keys={'self_attn.q_proj'})
    assert len(tree_flatten(model.trainable_parameters())) == 4


class ChatTokenizer:
    def __init__(self, family):
        self.chat_template = family
        self.formatted = ('<bos><start_of_turn>user\nraw<end_of_turn>\n<start_of_turn>model\n'
                          if family == 'gemma' else
                          '<|im_start|>user\nraw<|im_end|>\n<|im_start|>assistant\n')
        self.encodings = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{'role': 'user', 'content': 'raw'}]
        assert tokenize is False and add_generation_prompt is True
        return self.formatted

    def encode(self, text, *, add_special_tokens=True):
        assert text == self.formatted  # Completion text must never be re-encoded.
        self.encodings.append(add_special_tokens)
        return ([0] if add_special_tokens else []) + [0, 1]


@pytest.mark.parametrize('family', ['gemma', 'qwen'])
@pytest.mark.parametrize('finish_reason', ['stop', 'length'])
def test_sampling_and_scoring_share_prompt_and_exact_generated_ids(family, finish_reason):
    tokenizer = ChatTokenizer(family)
    prompts = []

    def stream(model, tokenizer, *, prompt, **kwargs):
        prompts.append(prompt)
        yield SimpleNamespace(text='hello', token=2, finish_reason=None)
        yield SimpleNamespace(text='' if finish_reason == 'stop' else '!',
                              token=3, finish_reason=finish_reason)

    with patch.object(models, 'stream_generate', side_effect=stream):
        text, ids = models.sample_group(None, tokenizer, 'raw', group_size=2)
    assert text == (['hello'] * 2 if finish_reason == 'stop' else ['hello!'] * 2)
    assert ids == [[2, 3], [2, 3]]
    assert prompts == [[0, 1], [0, 1]]

    scoring_inputs = []

    def scorer(inputs):
        scoring_inputs.append(inputs.tolist())
        return mx.zeros((1, inputs.shape[1], 4))

    assert float(models.sequence_logprobs(scorer, tokenizer, 'raw', ids[0])) == pytest.approx(-2 * math.log(4))
    assert scoring_inputs == [[[0, 1, 2]]]
    assert tokenizer.encodings == [False, False]


def test_base_checkpoint_and_empty_completion():
    tokenizer = BaseTokenizer()
    assert models.apply_chat_template(tokenizer, 'raw') == 'raw'
    assert models.encode_prompt(tokenizer, 'raw') == [0, 1]
    assert float(models.sequence_logprobs(None, tokenizer, 'raw', [])) == 0


@pytest.mark.parametrize('name', ['gemma', 'qwen'])
def test_cli_model_and_lora_options(name):
    from rl_framework.train import parse_args
    args = parse_args(['game_2048.RL2048', name, '--lora-rank', '4', '--lora-layers', '-1',
                       '--lora-keys', 'self_attn.q_proj'])
    assert (args.model_id, args.lora_rank, args.lora_layers, args.lora_keys) == (
        name, 4, -1, ['self_attn.q_proj'])


@pytest.mark.parametrize('module', [gemma3_text, qwen3])
def test_local_checkpoint_loading_generation_and_likelihood(tmp_path, module):
    """Exercise the real loader, HF tokenizer, and MLX-LM streaming implementation."""
    from tokenizers import Tokenizer, models as token_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from mlx_lm.sample_utils import make_sampler

    model = make_model(module)
    (tmp_path / 'config.json').write_text(json.dumps(asdict(model.args)))
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    special = ['<unk>', '<bos>', '<eos>', '<start_of_turn>', '<end_of_turn>',
               '<|im_start|>', '<|im_end|>']
    words = special + ['user', 'model', 'assistant', 'raw']
    words += [f'token{i}' for i in range(len(words), 64)]
    backend = Tokenizer(token_models.WordLevel(dict(zip(words, range(64))), unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    template = (
        "{{ bos_token }}<start_of_turn>user\n{{ messages[0]['content'] }}<end_of_turn>\n<start_of_turn>model\n"
        if module is gemma3_text else
        "<|im_start|>user\n{{ messages[0]['content'] }}<|im_end|>\n<|im_start|>assistant\n"
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                        bos_token='<bos>', eos_token='<eos>',
                                        additional_special_tokens=special[3:], chat_template=template)
    tokenizer.save_pretrained(tmp_path)

    model, tokenizer = models.setup_model(str(tmp_path), lora_rank=2, lora_layers=1)
    _, ids = models.sample_group(model, tokenizer, 'raw', group_size=1, max_tokens=3, temperature=0)
    assert 1 <= len(ids[0]) <= 3
    prompt_ids = models.encode_prompt(tokenizer, 'raw')
    assert prompt_ids.count(tokenizer.bos_token_id) == (1 if module is gemma3_text else 0)
    responses = list(models.stream_generate(model, tokenizer, prompt=prompt_ids, max_tokens=3,
                                            sampler=make_sampler(temp=0)))
    assert ids[0] == [response.token for response in responses]
    expected = sum(float(response.logprobs[response.token]) for response in responses)
    assert float(models.sequence_logprobs(model, tokenizer, 'raw', ids[0])) == pytest.approx(expected, abs=1e-4)
