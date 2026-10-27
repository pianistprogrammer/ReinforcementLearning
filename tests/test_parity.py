import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rl_framework.tasks import game_2048 as game
from rl_framework.tasks import mbpp_unittests as mbpp


def task_mbpp():
    rows = [{'task_id': 1, 'text': 'identity', 'test_list': ['assert f(1) == 1']}]
    with patch.object(mbpp, 'load_dataset', return_value=rows):
        return mbpp.MBPPUnitTests()


def test_raw_prompt():
    task = task_mbpp()
    class Tokenizer:
        def apply_chat_template(self, *a, **kw):
            raise AssertionError('task must return raw prompt')
    assert 'Task: identity' in task.make_prompt(Tokenizer())


def test_preview_and_actual_denominator():
    task = task_mbpp()
    assert task.preview()['train_count'] == 1
    with pytest.raises(PermissionError):
        task.evaluate(None, None)
    task.allow_unsafe_code = True
    with patch('rl_framework.core.models.sample_group', return_value=(['def f(x): return x'], [[]])):
        result = task.evaluate(None, None, n=50)
    assert result == {'passed': 1, 'evaluated': 1, 'pass_at_1': 1.0}
    task.eval = []
    assert task.evaluate(None, None)['evaluated'] == 0


def test_reward_guard():
    for task in (task_mbpp(), game.RL2048()):
        with pytest.raises(PermissionError):
            task.reward('def strategy(board): return "LEFT"')


def test_temp_cleanup(tmp_path):
    with patch.object(mbpp.tempfile, 'tempdir', str(tmp_path)):
        assert mbpp._run_tests('x=1', ['assert x == 1'], allow_unsafe_code=True)
        assert not mbpp._run_tests('while True: pass', [], timeout=.1, allow_unsafe_code=True)
    assert list(tmp_path.iterdir()) == []


def test_render_demo_and_partial_game():
    g = game.Game2048(seed=1)
    assert g.render().startswith('+')
    assert '<table' in game.show_board(g, '<unsafe>')
    assert '&lt;unsafe&gt;' in game.show_board(g, '<unsafe>')
    assert isinstance(game.demo(), game.Game2048)
    def broken(board):
        raise ValueError('oops')
    mt, score, board = game.run_strategy(broken, seed=1)
    assert mt == g.max_tile() and score == 0 and board.board == g.board


def test_game_exception_reward_and_timeout():
    task = game.RL2048(allow_unsafe_code=True, timeout=.5)
    reward, info = task.reward('def strategy(board):\n    raise ValueError("oops")')
    assert reward == .5 and info['avg_score'] == 0
    reward, info = task.reward('def strategy(board):\n    while True: pass')
    assert reward == 0 and info['reason'] == 'timeout'


def test_inspection_settings_and_final_board(tmp_path):
    task = game.RL2048(allow_unsafe_code=True)
    calls = []
    def sample(*a, **kw):
        calls.append(kw)
        return ['def strategy(board): return "LEFT"'] * 3, [[]] * 3
    with patch('rl_framework.core.models.sample_group', side_effect=sample):
        baseline = task.evaluate(None, None, phase='baseline')
        final = task.evaluate(None, None, phase='final', output_dir=tmp_path)
    assert calls[0] == {'group_size': 3, 'max_tokens': 2048, 'temperature': .7}
    assert calls[1] == {'group_size': 3, 'max_tokens': 512, 'temperature': .3}
    assert len(baseline) == len(final) == 3
    assert all('board' in result for result in final)
    assert len(list(tmp_path.glob('*.html'))) == 3


def test_cli_defaults_and_custom_tasks():
    from rl_framework.train import parse_args, load_task_class
    a = parse_args(['mbpp_unittests.MBPPUnitTests'])
    assert (a.steps, a.group, a.lr, a.max_tokens, a.temp) == (200, 6, 1e-5, 2048, .9)
    a = parse_args(['game_2048.RL2048', '--group', '9'])
    assert (a.steps, a.group, a.lr) == (200, 9, 5e-6)
    import types
    custom = types.ModuleType('custom_task')
    custom.Task = type('Task', (), {'defaults': {'group': 7}})
    with patch.dict(sys.modules, {'custom_task': custom}):
        assert parse_args(['custom_task:Task']).group == 7
        assert load_task_class('custom_task:Task') is custom.Task
    assert load_task_class('rl_framework.tasks.game_2048:RL2048') is game.RL2048


def test_training_function_count():
    task = game.RL2048()
    assert task.training_info(['def strategy(board): return "UP"', 'no function']) == 'functions 1/2'


def test_cli_guard_before_model_load():
    from rl_framework.train import main
    with patch('rl_framework.core.models.setup_model') as load:
        with pytest.raises(SystemExit, match='allow-unsafe-code'):
            main(['game_2048.RL2048'])
        load.assert_not_called()


def test_real_mlx_optimizer():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rl_framework.core.grpo import grpo_step
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.random.normal((4, 4))
        def __call__(self, ids):
            return self.lora_a[ids]
    class Tokenizer:
        def encode(self, text):
            return [0, 1]
    model, ref = Tiny(), Tiny()
    ref.update(model.parameters())
    ref.freeze()
    before = mx.array(model.lora_a)
    optimizer = optim.Adam(learning_rate=.01)
    loss = grpo_step(model, ref, Tokenizer(), optimizer, 'raw', ['a', 'b'], [[2], [3]], [0., 1.])
    assert math.isfinite(loss)
    assert bool(mx.any(before != model.lora_a))
    assert int(optimizer.step) == 1
