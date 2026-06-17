# pasqa

This repository contains materials developed by LY Corporation and is temporarily open-sourced for the purpose of a {paper URL}.

- **Temporary Release**: This repository is temporarily available as open-source. Therefore this repository may be turned into read-only or private anytime.
- **Attribution**: All code and materials in this repository are owned by LY Corporation.

## Project Overview

Minimal standalone inference package for the **AccentErrorMOS** SSLMOS model.

No dependency on the `sheet` training repository. Just point it at a checkpoint.

- Architecture: `SSLMOS` with `wav2vec2` (s3prl) backbone
- Mora cross-attention (`use_mora=True`) — mora token list is **required** at inference
- Frame-level error head (`use_error_head=True`) — outputs per-frame sigmoid probabilities
- Speaker GRL (`use_speaker_grl=True`) — included in model definition, inactive at inference

Vocab is resolved from the model's config (`mora_vocab_path`), walking up ancestor directories from the config file until the path is found.
`src/pasqa/vocab.txt` is bundled as a fallback when the config's vocab path cannot be located.

## Installation and Usage

```bash
cd /path/to/pasqa
uv sync
```

```python
from pasqa import PasqaPredictor

predictor = PasqaPredictor(
    checkpoint="path/to/exp/20260221_.../checkpoint-best.pkl",
    # config is auto-discovered from checkpoint directory (config.yml)
    # device defaults to 'cuda' if available
)

result = predictor.predict(
    wav_path="audio.wav",
    mora=["カ", "タ", "カ", "ナ"],
)
# result: {"mos": float, "frame_error_logits": np.ndarray, "frame_lengths": int}
print(result["mos"])

# Or with a raw tensor
import torch
result = predictor.predict(
    wav=torch.zeros(16000),
    mora=["ア", "イ", "ウ"],
)
```

## Acknowledgements

The model implementation is based on the [SHEET](https://github.com/unilight/sheet) toolkit.

## Contributions

As this project is temporarily open-sourced, we are not accepting contributions. For feedback or inquiries, please open an issue in this repository.

## License

This code is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). You may copy, modify, and distribute it without restriction, and the authors make no warranties or guarantees regarding its use.
