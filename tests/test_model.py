import torch

from src.aifusion.model import AIFusionLM, ModelConfig


def tiny_config():
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        dropout=0.0,
    )


def test_forward_shape_and_finite_loss():
    cfg = tiny_config()
    model = AIFusionLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    result = model(tokens, labels=tokens)
    assert result["logits"].shape == (2, 8, cfg.vocab_size)
    assert result["loss"] is not None
    assert torch.isfinite(result["loss"])


def test_backward_pass():
    cfg = tiny_config()
    model = AIFusionLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    result = model(tokens, labels=tokens)
    result["loss"].backward()
    assert model.token_embedding.weight.grad is not None
