"""Training entry point. Full useful training requires a real licensed corpus/tokenizer."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml

from src.aifusion.model import AIFusionLM, ModelConfig


def read_jsonl(path, text_key):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = item.get(text_key)
                if isinstance(text, str) and text.strip():
                    rows.append(text.strip())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = ModelConfig(**raw["model"])
    torch.manual_seed(int(raw.get("project", {}).get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AIFusionLM(cfg).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Trainable parameter count: {count:,}")
    print("Status: randomly initialized; not pretrained.")

    if args.smoke_test:
        # Tiny synthetic batch verifies forward/backward only; it is not training data.
        model.config.max_seq_len = min(model.config.max_seq_len, 16)
        batch = torch.randint(0, cfg.vocab_size, (1, model.config.max_seq_len), device=device)
        result = model(batch, labels=batch)
        result["loss"].backward()
        print(f"Smoke test passed. Synthetic loss: {result['loss'].item():.4f}")
        return

    data_cfg = raw["data"]
    train_path = Path(data_cfg["train_file"])
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data missing: {train_path}. Prepare licensed JSONL records "
            f"with a '{data_cfg['text_key']}' field first."
        )
    raise NotImplementedError(
        "Full training loop requires a trained tokenizer and packed token dataset. "
        "Do not launch long training until dataset, tokenizer, validation, and memory "
        "planning are complete."
    )


if __name__ == "__main__":
    main()
