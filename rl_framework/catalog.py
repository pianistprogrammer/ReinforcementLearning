"""Model names shared by the trainer and the lightweight web service."""
MODEL_PRESETS = {
    "gpt-oss": "mlx-community/gpt-oss-20b-MXFP4-Q4",
    "gemma": "mlx-community/gemma-3-1b-it-4bit",
    "qwen": "mlx-community/Qwen3-4B-4bit",
}
DEFAULT_MODEL_ID = MODEL_PRESETS["gpt-oss"]
