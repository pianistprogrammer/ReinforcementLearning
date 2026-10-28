"""Experimental GRPO-style training for MLX-LM compatible models."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
from rl_framework.events import emit, event_context


def load_task_class(spec):
    if ':' in spec:
        module, name = spec.rsplit(':', 1)
    else:
        module, name = spec.rsplit('.', 1)
        if '.' not in module:
            module = f'rl_framework.tasks.{module}'
    return getattr(importlib.import_module(module), name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', help='built-in module.Class or fully qualified module:Class')
    parser.add_argument('model_id', nargs='?', default='gpt-oss',
                        help='preset (gemma, qwen, gpt-oss), MLX repository ID, or local path')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-layers', type=int, default=8, help='last N layers; -1 for all')
    parser.add_argument('--lora-keys', nargs='+', default=None,
                        help='module paths within each layer (default: attention q/k/v/o projections)')
    parser.add_argument('--action', choices=['train', 'preview', 'demo', 'evaluate'], default='train')
    parser.add_argument('--phase', choices=['baseline', 'final'], default='final')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--group', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_tokens', '--max-tokens', dest='max_tokens', type=int)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--allow-unsafe-code', action='store_true', help='execute generated code WITHOUT a sandbox')
    parser.add_argument('--task-params', type=json.loads, default={},
                        help='JSON dict forwarded as kwargs to the task constructor')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args(argv)
    task_class = load_task_class(args.task)
    defaults = {'steps': 200, 'group': 6, 'lr': 5e-6, 'max_tokens': 2048, 'temp': .9}
    defaults.update(getattr(task_class, 'defaults', {}))
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv=None):
    args = parse_args(argv)
    emit('phase', phase='preparing', message='Preparing task', steps=args.steps)
    task_class = load_task_class(args.task)
    if args.action in ('train', 'evaluate') and getattr(task_class, 'executes_generated_code', False):
        if not args.allow_unsafe_code:
            raise SystemExit('Refusing generated code execution: pass --allow-unsafe-code. Subprocess is NOT a sandbox.')
        print('WARNING: executing generated code with your user privileges; subprocess is NOT a sandbox.')
    kwargs = {'allow_unsafe_code': args.allow_unsafe_code} if getattr(task_class, 'executes_generated_code', False) else {}
    if args.task_params:
        kwargs.update(args.task_params)
    task = task_class(**kwargs)
    if args.action == 'preview':
        result = task.preview()
        print(json.dumps(result, indent=2))
        emit('evaluation', phase='preview', result=result)
        return
    if args.action == 'demo':
        module = importlib.import_module(task_class.__module__)
        board = module.demo()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / 'demo.html'
        path.write_text(module.show_board(board, 'demo board'))
        print(board.render())
        print(path)
        emit('evaluation', phase='demo', result={'board': board.board, 'score': board.score})
        return

    from rl_framework.core.models import setup_model, sample_group
    model_options = {'lora_rank': args.lora_rank, 'lora_layers': args.lora_layers,
                     'lora_keys': args.lora_keys}
    model, tokenizer = setup_model(args.model_id, **model_options)

    def evaluate(phase):
        emit('phase', phase=phase, message=f'{phase.capitalize()} evaluation')
        with event_context(phase=phase):
            result = task.evaluate(model, tokenizer, phase=phase, output_dir=args.output_dir)
        print(f'{phase}: {json.dumps(result, indent=2)}')
        emit('evaluation', phase=phase, result=result)
        return result

    if args.action == 'evaluate':
        evaluate(args.phase)
        return

    import mlx.optimizers as optim
    from rl_framework.core.grpo import grpo_step
    emit('phase', phase='loading_reference', message='Loading frozen reference model')
    ref_model, _ = setup_model(args.model_id, **model_options)
    ref_model.freeze()
    optimizer = optim.Adam(learning_rate=args.lr)
    if hasattr(task, 'evaluate'):
        evaluate('baseline')
    enable_thinking = getattr(task, 'enable_thinking', None)
    for step in range(args.steps):
        with event_context(step=step + 1, phase='training'):
            emit('phase', phase='sampling', message=f'Sampling step {step + 1} / {args.steps}')
            prompt = task.make_prompt(tokenizer)
            completions, ids = sample_group(model, tokenizer, prompt, group_size=args.group,
                                            max_tokens=args.max_tokens, temperature=args.temp,
                                            enable_thinking=enable_thinking)
            emit('phase', phase='rewarding', message='Evaluating completions')
            rewards = [task.reward(completion)[0] for completion in completions]
            emit('phase', phase='updating', message='Updating LoRA adapters')
            loss = grpo_step(model, ref_model, tokenizer, optimizer, prompt, completions, ids, rewards,
                             enable_thinking=enable_thinking)
            emit('metric', loss=loss, mean_reward=sum(rewards) / len(rewards),
                 rewards=rewards, steps=args.steps)
        if step % 10 == 0:
            detail = f' | {task.training_info(completions)}' if hasattr(task, 'training_info') else ''
            print(f'step {step:3d} | loss {loss:.4f} | mean reward {sum(rewards) / len(rewards):.3f}{detail}')
    if hasattr(task, 'evaluate'):
        evaluate('final')


if __name__ == '__main__':
    main()
