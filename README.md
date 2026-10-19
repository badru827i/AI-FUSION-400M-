# AI-FUSION-400M

An experimental language model being developed from random initialization for AI-FUSION Assistant. Target capabilities: Bahasa Malaysia + English, structured CAD/Text-to-3D instructions, and offline Android inference.

## Status: early scaffold

- [x] Repository and initial decoder-only Transformer implementation
- [x] Initial configuration and synthetic forward/backward smoke test
- [ ] Tokenizer training and versioning
- [ ] Licensed corpus collection, filtering, deduplication, and splits
- [ ] Complete real-data training loop
- [ ] Instruction/CAD tuning and held-out evaluation
- [ ] ONNX export and graph validation
- [ ] INT4 quantization and quality/performance benchmarks
- [ ] Android offline runtime integration

**Important:** this repository does not yet contain trained model weights. The current code is randomly initialized and is not an already-trained AI model. A smoke test only checks that code executes; it does not demonstrate language ability.

## Quick start

Python 3.11+ recommended.

```bash
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
python -m src.aifusion.train --config configs/model.yaml --smoke-test
```

The smoke test creates synthetic token IDs and runs a forward/backward pass. It does not use or create a useful trained model.

## Dataset format

The first training data format is JSON Lines, one JSON object per line:

```json
{"text": "Bina objek berbentuk kubus dengan sisi 20 mm."}
{"text": "Create a symmetric mechanical bracket with four mounting holes."}
```

These examples document the schema only. They are not a sufficient training corpus. Keep train/validation/test files separate, audit source licenses and provenance, remove duplicates, and never put private or sensitive text into the corpus.

## Model lifecycle

Random initialization → pretraining → instruction/CAD tuning → evaluation → ONNX export → INT4 quantization → Android runtime tests.

The language model should emit constrained, structured geometry instructions. A separate deterministic geometry engine must validate parameters and create the mesh. The language model alone cannot guarantee valid OBJ output.

## Training guidance

Do not launch long runs until the model parameter count, optimizer memory, sequence length, dataset, tokenizer, validation metrics, and available GPU memory have been checked. Start with a small smoke test, then a small real-data overfit test, then short measured runs before scaling up.

Never commit large datasets, checkpoints, access tokens, or secrets to Git.
