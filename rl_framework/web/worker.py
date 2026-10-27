"""Subprocess entry point. Stdout carries events; stderr carries ordinary logs."""
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

from rl_framework.events import emit, event_context, event_sink
from .config import RunConfig

PREFIX = '@rl:'


def demo(config):
    """A labeled transport demonstration; never loads a model or executes code."""
    emit('phase', phase='preparing', message='Demo stream · no model or generated-code execution')
    print('Demo stream connected. Metrics and completions below are simulated.', file=sys.stderr, flush=True)
    for step in range(1, min(config.steps, 12) + 1):
        with event_context(step=step, phase='training'):
            emit('phase', phase='sampling', message=f'Demo sampling · step {step}')
            generation_id = f'demo-{step}'
            emit('generation_start', generation_id=generation_id, sample_index=0, group_size=1,
                 prompt='Write a short Python strategy that plays 2048.' if config.task.startswith('game') else 'Write a Python identity function.')
            text = ('def strategy(board):\n    # Prefer moves that keep the largest tile in a corner.\n'
                    '    empty = sum(cell == 0 for row in board for cell in row)\n'
                    '    return "LEFT" if empty > 4 else "DOWN"\n' if config.task.startswith('game') else
                    'def identity(value):\n    """Return the input unchanged."""\n    return value\n')
            for index, offset in enumerate(range(0, len(text), 5)):
                emit('token', generation_id=generation_id, text=text[offset:offset+5], token_id=index,
                     tokens=index+1, tokens_per_second=31.5 + step * .7, peak_memory_gb=None)
                time.sleep(.025)
            emit('generation_end', generation_id=generation_id, tokens=index+1)
            emit('phase', phase='updating', message='Demo adapter update')
            time.sleep(.15)
            emit('metric', loss=1.8 / (1 + step * .24), mean_reward=min(1.8, .22 + step * .12),
                 rewards=[.22 + step * .12], steps=min(config.steps, 12))
            print(f'DEMO step {step:3d} | simulated loss {1.8 / (1 + step * .24):.4f}', file=sys.stderr, flush=True)
    emit('phase', phase='final', message='Demo stream finished')


def main():
    payload = json.loads(sys.stdin.readline())
    config = RunConfig.parse(payload['config'])
    wire = sys.stdout

    def send(event):
        wire.write(PREFIX + json.dumps(event, allow_nan=False) + '\n')
        wire.flush()

    with event_sink(send), redirect_stdout(sys.stderr):
        if config.mode == 'demo':
            demo(config)
        else:
            from rl_framework.train import main as train
            train(config.argv(Path(payload['output_dir'])))


if __name__ == '__main__':
    main()
