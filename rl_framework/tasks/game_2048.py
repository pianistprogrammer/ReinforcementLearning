# tasks/game_2048.py — reward: code that plays 2048 well
from __future__ import annotations
import re, random
import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

from .base import RLTask


def _extract_function(text: str):
    m = re.search(r"```(?:python)?\s*(def strategy.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(def strategy\(board\):.*?)(?=\n\n|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _compress_merge(row):
    vals = [v for v in row if v != 0]
    merged, gained, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i+1]:
            merged.append(vals[i]*2); gained += vals[i]*2; i += 2
        else:
            merged.append(vals[i]); i += 1
    merged += [0] * (4 - len(merged))
    return merged, gained


class Game2048:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [[0]*4 for _ in range(4)]
        self.score = 0
        self._spawn(); self._spawn()

    def _spawn(self):
        empties = [(r,c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if empties:
            r, c = self.rng.choice(empties)
            self.board[r][c] = 4 if self.rng.random() < 0.1 else 2

    def move(self, direction: str):
        b = [row[:] for row in self.board]
        rot = {"LEFT":0,"UP":1,"RIGHT":2,"DOWN":3}[direction.upper()]
        arr = np.rot90(np.array(b), rot)
        gained_total, new_rows = 0, []
        for row in arr:
            merged, gained = _compress_merge(row.tolist())
            new_rows.append(merged); gained_total += gained
        arr = np.rot90(np.array(new_rows), -rot)
        new_board = arr.tolist()
        moved = new_board != self.board
        if moved:
            self.board = new_board
            self.score += gained_total
            self._spawn()
        return moved

    def game_over(self):
        if any(self.board[r][c] == 0 for r in range(4) for c in range(4)):
            return False
        for d in ["LEFT","RIGHT","UP","DOWN"]:
            test = Game2048.__new__(Game2048)
            test.board = [row[:] for row in self.board]
            test.score = 0; test.rng = random.Random()
            if test.move(d): return False
        return True

    def max_tile(self):
        return max(max(r) for r in self.board)

    def render(self):
        width = max(4, len(str(self.max_tile())))
        sep = '+' + ('-' * width + '+') * 4
        lines = [sep]
        for row in self.board:
            lines.append('|' + '|'.join(str(v).center(width) if v else ' ' * width for v in row) + '|')
            lines.append(sep)
        return '\n'.join(lines)


class RL2048(RLTask):
    executes_generated_code = True
    defaults = {'steps': 200, 'group': 4, 'lr': 5e-6, 'max_tokens': 2048, 'temp': .9}

    def __init__(self, *, allow_unsafe_code=False, timeout=5, enable_thinking=False):
        self.allow_unsafe_code = allow_unsafe_code
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    def require_execution(self):
        if not self.allow_unsafe_code:
            raise PermissionError('Generated code execution requires --allow-unsafe-code; subprocess is not a sandbox.')

    def make_prompt(self, tokenizer=None) -> str:
        return """Write a short Python function that plays 2048.

- Function signature: def strategy(board):
- board is a list of 4 lists of 4 ints (0 = empty)
- Return exactly one of: "UP", "DOWN", "LEFT", "RIGHT"
- Use only Python standard library (math, random, copy ok — no numpy)
- All helpers must be inside strategy()
- Keep it concise, under 30 lines

Output ONLY the code block, no explanation:
```python
def strategy(board):
    # implementation
    return "UP"
```"""

    def training_info(self, completions):
        count = sum(_extract_function(completion) is not None for completion in completions)
        return f'functions {count}/{len(completions)}'

    def reward(self, completion):
        result = self.execute(completion)
        return result['reward'], result['info']

    def execute(self, completion, *, final=False):
        self.require_execution()
        project_root = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.json'
            response = Path(directory) / 'response.json'
            request.write_text(json.dumps({'completion': completion, 'final': final}))
            try:
                subprocess.run([os.path.abspath(sys.executable), '-m', 'rl_framework.tasks.game_2048',
                                str(request), str(response)], timeout=self.timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                               cwd=project_root, env=env)
                return json.loads(response.read_text())
            except subprocess.TimeoutExpired:
                return {'reward': 0.0, 'info': {'reason': 'timeout'}}
            except (subprocess.CalledProcessError, OSError, ValueError):
                return {'reward': 0.0, 'info': {'reason': 'worker_error'}}

    def evaluate(self, model, tokenizer, *, phase='final', output_dir=None, **kwargs):
        self.require_execution()
        from rl_framework.core.models import sample_group
        baseline = phase == 'baseline'
        completions, _ = sample_group(model, tokenizer, self.make_prompt(tokenizer),
                                      group_size=3, max_tokens=2048 if baseline else 512,
                                      temperature=.7 if baseline else .3,
                                      enable_thinking=self.enable_thinking)
        results = []
        for index, completion in enumerate(completions):
            result = self.execute(completion, final=not baseline)
            result['raw'] = completion[:300]
            source = _extract_function(completion)
            result['function'] = source[:300 if baseline else 400] if source else None
            if 'board' in result:
                board = Game2048(seed=99)
                board.board, board.score = result['board'], result['score']
                result['ascii'] = board.render()
                if output_dir is not None:
                    path = Path(output_dir) / f'final-{index + 1}.html'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(show_board(board, 'trained strategy'))
                    result['html'] = str(path)
            results.append(result)
        return results


def run_strategy(strategy_fn, seed=None, max_moves=400):
    """Run a trusted callable; generated source must go through RL2048.execute."""
    game = Game2048(seed=seed)
    for _ in range(max_moves):
        if game.game_over():
            break
        try:
            move = strategy_fn([row[:] for row in game.board])
            if move not in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                break
            game.move(move)
        except Exception:
            break
    return game.max_tile(), game.score, game


COLORS = {
    0: ('#cdc1b4', '#776e65'), 2: ('#eee4da', '#776e65'), 4: ('#ede0c8', '#776e65'),
    8: ('#f2b179', '#f9f6f2'), 16: ('#f59563', '#f9f6f2'), 32: ('#f67c5f', '#f9f6f2'),
    64: ('#f65e3b', '#f9f6f2'), 128: ('#edcf72', '#f9f6f2'), 256: ('#edcc61', '#f9f6f2'),
    512: ('#edc850', '#f9f6f2'), 1024: ('#edc53f', '#f9f6f2'), 2048: ('#edc22e', '#f9f6f2'),
}


def show_board(game, label=''):
    rows = []
    for row in game.board:
        cells = []
        for value in row:
            bg, fg = COLORS.get(value, ('#3c3a32', '#f9f6f2'))
            size = '2em' if value < 100 else '1.5em' if value < 1000 else '1.1em'
            cells.append(f'<td style="background:{bg};color:{fg};width:80px;height:80px;'
                         f'text-align:center;font-size:{size};font-weight:bold;border-radius:6px">'
                         f'{value if value else ""}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>2048</title></head>'
            f'<body><p>Score: {game.score} | {html.escape(label)}</p>'
            '<table style="border-spacing:6px;background:#bbada0;padding:6px;border-radius:8px">'
            + ''.join(rows) + '</table></body></html>')


def demo():
    game = Game2048(seed=42)
    rng = random.Random(42)
    for _ in range(20):
        game.move(rng.choice(['UP', 'DOWN', 'LEFT', 'RIGHT']))
    return game


def _worker(completion, final=False):
    source = _extract_function(completion)
    if source is None:
        return {'reward': -1.0, 'info': {'reason': 'no_function'}}
    try:
        namespace = {}
        exec(compile(source, '<strategy>', 'exec'), {'__builtins__': __builtins__}, namespace)
        strategy = namespace.get('strategy')
        if strategy is None:
            return {'reward': -.5, 'info': {'reason': 'no_strategy_name'}}
    except Exception as error:
        return {'reward': -.5, 'info': {'reason': f'compile_error: {error}'}}
    games = [run_strategy(strategy, seed=seed) for seed in (1, 2, 3)]
    average = sum(score for _, score, _ in games) / len(games)
    best = max(tile for tile, _, _ in games)
    reward = .5 + min(.5, average / 2000)
    for threshold, bonus in ((256, .2), (512, .3), (1024, .5), (2048, 1.)):
        if best >= threshold:
            reward += bonus
    result = {'reward': min(2., reward), 'info': {'avg_score': average, 'best_tile': best}}
    if final:
        tile, score, game = run_strategy(strategy, seed=99, max_moves=500)
        result.update(board=game.board, score=score, max_tile=tile)
    return result


if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(_worker(**request)))
