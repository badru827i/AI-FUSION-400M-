"""Training entry point. Full useful training requires a real licensed corpus/tokenizer."""
import argparse
import json
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
    torch.manual_seed(int(raw.get("project", {}).get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke_test:
        # Small synthetic model avoids allocating the full target architecture in CI.
        cfg = ModelConfig(
            vocab_size=128, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=128, max_seq_len=16, dropout=0.0,
        )
        model = AIFusionLM(cfg).to(device)
        batch = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len), device=device)
        result = model(batch, labels=batch)
        if result["loss"] is None or not torch.isfinite(result["loss"]):
            raise RuntimeError("Smoke test produced an invalid loss.")
        result["loss"].backward()
        print(f"Device: {device}")
        print("Smoke test passed with a tiny synthetic model.")
        print("This does not train or validate the target model.")
        return

    cfg = ModelConfig(**raw["model"])
    model = AIFusionLM(cfg).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Trainable parameter count: {count:,}")
    print("Status: randomly initialized; not pretrained.")

    data_cfg = raw["data"]
    train_path = Path(data_cfg["train_file"])
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data missing: {train_path}. Prepare licensed JSONL records "
            f"with a '{data_cfg['text_key']}' field first."
        )
    raise NotImplementedError(
        "Full training loop is not implemented yet. A tokenizer, packed token "
        "dataset, validation pipeline, optimizer/memory plan, and measured GPU "
        "capacity are required before long training."
    )


if __name__ == "__main__":
    main()
