"""Validated browser inputs; custom Python imports remain a CLI-only feature."""
from dataclasses import asdict, dataclass, fields
import math

TASKS = {
    "game_2048.RL2048": {"label": "2048 strategy", "steps": 200, "group": 4, "lr": 5e-6},
    "mbpp_unittests.MBPPUnitTests": {"label": "MBPP unit tests", "steps": 200, "group": 6, "lr": 1e-5},
}


@dataclass
class RunConfig:
    task: str = "game_2048.RL2048"
    model_id: str = "gemma"
    action: str = "train"
    mode: str = "training"
    steps: int = 200
    group: int = 4
    lr: float = 5e-6
    max_tokens: int = 512
    temp: float = .9
    lora_rank: int = 8
    lora_layers: int = 8
    allow_unsafe_code: bool = False

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Run settings must be an object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        config = cls(**data)
        if not isinstance(config.task, str) or config.task not in TASKS:
            raise ValueError("Choose a supported task")
        if config.action not in ("train", "evaluate", "preview", "demo"):
            raise ValueError("Invalid run action")
        if config.mode not in ("training", "demo"):
            raise ValueError("Invalid run mode")
        if config.action == "preview" and config.task != "mbpp_unittests.MBPPUnitTests":
            raise ValueError("Dataset preview is available for MBPP")
        if config.action == "demo" and config.task != "game_2048.RL2048":
            raise ValueError("Board demo is available for 2048")
        if (not isinstance(config.model_id, str) or not config.model_id.strip()
                or len(config.model_id) > 512 or any(ord(c) < 32 for c in config.model_id)):
            raise ValueError("Enter a model preset, repository ID, or local path")
        for key, low, high in [("steps", 1, 2000), ("group", 1, 16), ("max_tokens", 1, 8192),
                               ("lora_rank", 1, 128), ("lora_layers", -1, 512)]:
            value = getattr(config, key)
            if type(value) is not int or not low <= value <= high or (key == "lora_layers" and value == 0):
                raise ValueError(f"Invalid {key}: expected {low} to {high}" + (", excluding 0" if key == 'lora_layers' else ""))
        for key, low, high in [("lr", 0, 1), ("temp", 0, 2)]:
            value = getattr(config, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}")
        if config.lr == 0:
            raise ValueError("Learning rate must be positive")
        if type(config.allow_unsafe_code) is not bool:
            raise ValueError("allow_unsafe_code must be a boolean")
        if config.mode == "training" and config.action in ("train", "evaluate") and not config.allow_unsafe_code:
            raise ValueError("Enable generated-code execution to run this task, or try the demo stream")
        return config

    def to_dict(self):
        return asdict(self)

    def argv(self, output_dir):
        args = [self.task, self.model_id, "--action", self.action, "--output-dir", str(output_dir)]
        for key in ("steps", "group", "lr", "max_tokens", "temp", "lora_rank", "lora_layers"):
            args += ["--" + key.replace('_', '-'), str(getattr(self, key))]
        if self.allow_unsafe_code:
            args.append("--allow-unsafe-code")
        return args
