from __future__ import annotations
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from datasets import load_dataset


def _extract_code(completion: str) -> str:
    completion = re.sub(r"<\|channel>thought.*?<\|/channel>", "", completion, flags=re.DOTALL)
    if "```python" in completion:
        completion = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        completion = completion.split("```")[1].split("```")[0]
    return completion.strip()


def _run_tests(code, tests, timeout=5, *, allow_unsafe_code=False):
    if not allow_unsafe_code:
        raise PermissionError('Generated code execution requires --allow-unsafe-code; subprocess is not a sandbox.')
    project_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'candidate.py'
        path.write_text(code + '\n\n' + '\n'.join(tests))
        try:
            result = subprocess.run([os.path.abspath(sys.executable), str(path)],
                                    capture_output=True, timeout=timeout,
                                    cwd=project_root, env=env)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


class MBPPUnitTests:
    executes_generated_code = True
    defaults = {'steps': 200, 'group': 6, 'lr': 1e-5, 'max_tokens': 2048, 'temp': .9}

    def __init__(self, split_train='train[:200]', split_eval='test[:50]', *, allow_unsafe_code=False):
        self.allow_unsafe_code = allow_unsafe_code
        self.ds = load_dataset('google-research-datasets/mbpp', split=split_train)
        self.eval = load_dataset('google-research-datasets/mbpp', split=split_eval)
        self._row_iter = iter(self.ds)

    def require_execution(self):
        if not self.allow_unsafe_code:
            raise PermissionError('Generated code execution requires --allow-unsafe-code; subprocess is not a sandbox.')

    def next_row(self):
        try:
            return next(self._row_iter)
        except StopIteration:
            self._row_iter = iter(self.ds)
            return next(self._row_iter)

    @staticmethod
    def prompt_for_row(row):
        return (f"Write a Python function.\n\nTask: {row['text']}\n\n"
                'Return ONLY the Python function code, no explanation, no markdown prose.')

    def make_prompt(self, tokenizer=None):
        self._current = self.next_row()
        return self.prompt_for_row(self._current)

    def preview(self, n=10):
        return {'train_count': len(self.ds), 'eval_count': len(self.eval),
                'rows': [{key: row[key] for key in ('task_id', 'text', 'test_list')}
                         for row in list(self.ds)[:n]]}

    def reward(self, completion):
        self.require_execution()
        row = getattr(self, '_current', None)
        if row is None:
            return -1.0, {'reason': 'no_active_row'}
        code = _extract_code(completion)
        if not code:
            return -1.0, {'reason': 'no_code'}
        passed = sum(_run_tests(code, [test], allow_unsafe_code=True) for test in row['test_list'])
        fraction = passed / max(len(row['test_list']), 1)
        return (-1.0 if passed == 0 else fraction * 2 - 1), {'passed': passed, 'total': len(row['test_list'])}

    def evaluate(self, model, tokenizer, n=50, **kwargs):
        self.require_execution()
        from rl_framework.core.models import sample_group
        passed = evaluated = 0
        for row in list(self.eval)[:max(0, n)]:
            completions, _ = sample_group(model, tokenizer, self.prompt_for_row(row),
                                          group_size=1, max_tokens=512, temperature=.2)
            passed += _run_tests(_extract_code(completions[0]), row['test_list'], allow_unsafe_code=True)
            evaluated += 1
        return {'passed': passed, 'evaluated': evaluated,
                'pass_at_1': passed / evaluated if evaluated else 0.0}
