# Dataset preparation

The repository intentionally does not download or bundle a training corpus.

## Record format

Use UTF-8 JSONL with one record per line and a non-empty `text` field:

```json
{"text": "Bina sebuah kubus simetri dengan sisi 20 mm."}
{"text": "Create a mechanical bracket with four mounting holes."}
```

## Required checks

1. Record source URL/name, collection date, license, language, and permitted use in a separate manifest.
2. Exclude private, sensitive, or unlawfully obtained data.
3. Normalize Unicode and whitespace consistently.
4. Deduplicate exact and near-duplicate records.
5. Split by source/document before creating train, validation, and test sets to reduce leakage.
6. Validate generated CAD/geometry examples with a parser and deterministic mesh checks.
7. Keep held-out evaluation prompts out of training and instruction-tuning sets.

The proposed bilingual proportions are an experiment, not a requirement. Evaluate Malay and English separately and adjust the mixture based on held-out quality.

## Text-to-3D examples

Prefer structured targets with explicit units, coordinate frame, primitive/component list, symmetry, constraints, and validation expectations. Do not treat arbitrary OBJ text as valid without parsing it.
