# PASQA

This repository contains materials developed by LY Corporation and is temporarily open-sourced for the purpose of a [paper](https://arxiv.org/abs/2606.20137).

Accepted to INTERSPEECH 2026

- **Temporary Release**: This repository is temporarily available as open-source. Therefore this repository may be turned into read-only or private anytime.
- **Attribution**: All code and materials in this repository are owned by LY Corporation.

## Project Overview

**PASQA (Pitch-Accent-focused Speech Quality Assessment)** is a mean opinion score (MOS)
prediction model that explicitly targets pitch-accent correctness. It is trained on a
controlled Japanese accent-error dataset, constructed by
changing accent patterns using an accent-controllable text-to-speech system, with a pseudo
accent-quality score computed from the accent-error rate. PASQA builds on self-supervised
representations and employs mora-conditioned fusion, ranking loss, an auxiliary accent-error
localization task, and speaker-invariant training.

## Installation and Usage

### 1. Install

```bash
uv sync
```

### 2. Download the pretrained model
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-PASQA-yellow)](https://huggingface.co/ly-corporation/PASQA)

Download the checkpoint `.pkl` and its `config.yml` from [Hugging Face](https://huggingface.co/ly-corporation/PASQA) and place them together anywhere you like. 
The layout below (e.g. a `pretrained/` directory) is only an example:

```
pasqa/
├── pretrained/
│   ├── checkpoint-100000steps.pkl   # model weights
│   └── config.yml            # auto-discovered from the checkpoint's directory
└── src/pasqa/vocab.txt       # bundled mora vocab (fallback)
```

The mora vocab is resolved from the config's `mora_vocab_path`; if that path cannot be
found, the bundled `src/pasqa/vocab.txt` is used automatically.

### 3. Run inference

```python
from pasqa import PasqaPredictor

predictor = PasqaPredictor(
    checkpoint="pretrained/checkpoint-best.pkl",
    # config is auto-discovered from the checkpoint's directory (config.yml)
    # device defaults to 'cuda' if available, else 'cpu'
)

result = predictor.predict(
    wav_path="audio.wav",
    mora=["ア", "シ", "タ", "ノ", "テ", "ン", "キ", "ハ", "ハ", "レ", "デ", "ス"],  # katakana mora list is REQUIRED
)

print(result["mos"])  # predicted MOS, ~1–5
```

## Acknowledgements

The model implementation is based on the [SHEET](https://github.com/unilight/sheet) toolkit.

## Contributions

As this project is temporarily open-sourced, we are not accepting contributions. For feedback or inquiries, please open an issue in this repository.

## License

This code is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). You may copy, modify, and distribute it without restriction, and the authors make no warranties or guarantees regarding its use.
