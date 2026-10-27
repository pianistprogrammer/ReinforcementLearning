"""Generic task adapter for any Hugging Face dataset or local CSV / JSONL file."""
from __future__ import annotations
from pathlib import Path


def _token_f1(pred: str, target: str) -> float:
    p_toks = set(pred.lower().split())
    t_toks = set(target.lower().split())
    if not p_toks or not t_toks:
        return 0.0
    common = p_toks & t_toks
    if not common:
        return 0.0
    precision = len(common) / len(p_toks)
    recall = len(common) / len(t_toks)
    return 2 * precision * recall / (precision + recall)


class HFGenericTask:
    executes_generated_code = False
    defaults = {'steps': 200, 'group': 6, 'lr': 1e-5, 'max_tokens': 512, 'temp': .9}

    def __init__(self, dataset_id: str, question_col: str, answer_col: str,
                 reward_type: str = 'contains', split: str = 'train',
                 eval_split: str = '', reward_module: str = '', **_):
        from datasets import load_dataset
        self.question_col = question_col
        self.answer_col = answer_col
        self.reward_type = reward_type

        path = Path(dataset_id)
        if path.is_absolute() and path.exists():
            fmt = 'csv' if path.suffix.lower() == '.csv' else 'json'
            self.ds = load_dataset(fmt, data_files=str(path), split='train')
            self.eval_ds = None
        else:
            self.ds = load_dataset(dataset_id, split=split or 'train', trust_remote_code=False)
            self.eval_ds = None
            if eval_split:
                try:
                    self.eval_ds = load_dataset(dataset_id, split=eval_split, trust_remote_code=False)
                except Exception:
                    pass

        self._custom_reward = None
        if reward_type == 'custom':
            import importlib.util
            spec = importlib.util.spec_from_file_location('_hf_reward', reward_module)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not callable(getattr(mod, 'reward', None)):
                raise ValueError(f"{reward_module} must define a callable reward(completion, row) -> float")
            self._custom_reward = mod.reward

        self._row_iter = iter(self.ds)
        self._current: dict | None = None

    def _next_row(self) -> dict:
        try:
            row = next(self._row_iter)
        except StopIteration:
            self._row_iter = iter(self.ds)
            row = next(self._row_iter)
        self._current = dict(row)
        return self._current

    def make_prompt(self, tokenizer) -> str:
        row = self._next_row()
        question = str(row[self.question_col])
        return f"Answer the following question concisely.\n\nQuestion: {question}\n\nAnswer:"

    def reward(self, completion: str):
        if self._current is None:
            return 0.0, {}
        answer = str(self._current[self.answer_col])
        c = completion.strip()

        if self.reward_type == 'exact_match':
            score = 1.0 if c.lower() == answer.lower() else -1.0
        elif self.reward_type == 'contains':
            score = 1.0 if answer.lower() in c.lower() else -1.0
        elif self.reward_type == 'f1_overlap':
            score = _token_f1(c, answer) * 2.0 - 1.0  # map [0,1] → [-1,1]
        elif self.reward_type == 'regex':
            import re
            try:
                score = 1.0 if re.search(answer, c, re.IGNORECASE) else -1.0
            except re.error:
                score = 0.0
        elif self.reward_type == 'custom':
            if self._custom_reward is not None:
                try:
                    score = float(self._custom_reward(c, dict(self._current)))
                    score = max(-2.0, min(2.0, score))
                except Exception as exc:
                    score = -1.0
                    return score, {'error': str(exc), 'reward_type': 'custom'}
            else:
                score = 0.0
        else:
            score = 0.0

        return score, {'answer': answer[:80], 'reward_type': self.reward_type}

    def training_info(self, completions: list) -> str:
        if self._current is None:
            return ''
        return f"answer: {str(self._current[self.answer_col])[:60]!r}"

    def evaluate(self, model, tokenizer, n: int = 50, **kwargs):
        from rl_framework.core.models import sample_group
        import re
        ds = self.eval_ds if self.eval_ds is not None else self.ds
        rows = list(ds)[:n]
        correct = 0
        for row in rows:
            question = str(row[self.question_col])
            answer = str(row[self.answer_col])
            prompt = f"Answer the following question concisely.\n\nQuestion: {question}\n\nAnswer:"
            completions, _ = sample_group(model, tokenizer, prompt, group_size=1,
                                          max_tokens=256, temperature=0.0)
            c = completions[0].strip() if completions else ''
            if self.reward_type == 'exact_match':
                correct += int(c.lower() == answer.lower())
            elif self.reward_type == 'contains':
                correct += int(answer.lower() in c.lower())
            elif self.reward_type == 'f1_overlap':
                correct += int(_token_f1(c, answer) > 0.5)
            elif self.reward_type == 'regex':
                try:
                    correct += int(bool(re.search(answer, c, re.IGNORECASE)))
                except re.error:
                    pass
            elif self.reward_type == 'custom' and self._custom_reward is not None:
                try:
                    correct += int(float(self._custom_reward(c, dict(row))) > 0)
                except Exception:
                    pass
        total = len(rows)
        return {
            'accuracy': correct / total if total else 0.0,
            'n': total,
            'reward_type': self.reward_type,
            'split': 'eval' if self.eval_ds is not None else 'train',
        }

    def preview(self, n: int = 10) -> dict:
        def _safe(v):
            return str(v)[:200] if not isinstance(v, (str, int, float, bool, type(None))) else v

        rows = [{k: _safe(v) for k, v in dict(row).items()} for row in list(self.ds)[:n]]
        return {
            'train_count': len(self.ds),
            'eval_count': len(self.eval_ds) if self.eval_ds is not None else 0,
            'columns': list(self.ds.column_names),
            'rows': rows,
        }
