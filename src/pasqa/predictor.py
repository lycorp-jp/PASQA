#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PasqaPredictor — standalone inference for the AccentErrorMOS SSLMOS model."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import yaml


class PasqaPredictor:
    """Standalone predictor for the AccentErrorMOS SSLMOS model.

    Args:
        checkpoint: Path to checkpoint .pkl file.
        config: Path to config.yml. If None, looks for config.yml in the
                same directory as the checkpoint.
        device: torch device string. Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        config: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        checkpoint = Path(checkpoint)
        # For config discovery, use the directory of the original (symlink) path
        checkpoint_dir = checkpoint.absolute().parent
        # Resolve symlinks, including broken relative ones (the symlink target may be
        # relative to an ancestor directory rather than the symlink's own directory).
        checkpoint = self._resolve_checkpoint(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        # Resolve config — prefer the original checkpoint's directory (e.g. exp dir)
        if config is None:
            config = checkpoint_dir / "config.yml"
            if not config.exists():
                config = checkpoint.parent / "config.yml"
        config = Path(config).resolve()
        if not config.exists():
            raise FileNotFoundError(f"Config not found: {config}")

        with open(config) as f:
            cfg = yaml.safe_load(f)

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Build vocab mapping — prefer mora_vocab_path from config, fall back to bundled
        vocab_path = self._resolve_vocab_path(cfg.get("mora_vocab_path"), config)
        self.mora_vocab = self._load_vocab(vocab_path)

        # Resolve mora_vocab_size
        model_params = dict(cfg.get("model_params", {}))
        if not model_params.get("mora_vocab_size"):
            model_params["mora_vocab_size"] = len(self.mora_vocab)

        # Resolve num_speakers
        if "num_speakers" in cfg:
            model_params.setdefault("num_speakers", cfg["num_speakers"])

        # Instantiate model
        from pasqa._sheet.model import SSLMOS

        model_params["model_input"] = cfg.get("model_input", "waveform")
        self.model = SSLMOS(**model_params)

        # Load checkpoint (follow symlinks)
        ckpt_real = checkpoint.resolve()
        state = torch.load(ckpt_real, map_location="cpu")
        # Support both raw state_dict and {"model": state_dict} wrappers
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_checkpoint(path: Path) -> Path:
        """Resolve a checkpoint path, handling broken relative symlinks.

        Some checkpoints are symlinks whose target is relative to an ancestor
        directory rather than the symlink's own directory. This method tries
        each ancestor until it finds an existing target.
        """
        path = path.absolute()
        if not path.is_symlink():
            return path
        if path.exists():
            return path.resolve()
        # Broken symlink — try resolving target relative to each ancestor
        target = os.readlink(str(path))
        if os.path.isabs(target):
            return Path(target)
        candidate_dir = path.parent
        while True:
            candidate = (candidate_dir / target).resolve()
            if candidate.exists():
                return candidate
            parent = candidate_dir.parent
            if parent == candidate_dir:
                break
            candidate_dir = parent
        # Fall back to standard resolve (will still fail, but gives a clear path)
        return path.resolve()

    @staticmethod
    def _resolve_vocab_path(
        mora_vocab_path: Optional[str], config: Path
    ) -> Path:
        """Resolve mora_vocab_path from config.

        If mora_vocab_path is an absolute path, use it directly.
        If relative, walk up ancestor directories of the config file until found.
        Falls back to the bundled vocab.txt if nothing is found.
        """
        bundled = Path(__file__).parent / "vocab.txt"
        if not mora_vocab_path:
            return bundled

        p = Path(mora_vocab_path)
        if p.is_absolute():
            return p if p.exists() else bundled

        # Relative path: try each ancestor directory of the config
        candidate_dir = config.parent
        while True:
            candidate = candidate_dir / p
            if candidate.exists():
                return candidate
            parent = candidate_dir.parent
            if parent == candidate_dir:
                break
            candidate_dir = parent

        # Also try just the filename alongside the config (e.g. pretrained/vocab.txt)
        alongside = config.parent / p.name
        if alongside.exists():
            return alongside

        return bundled

    @staticmethod
    def _load_vocab(path: Path) -> Dict[str, int]:
        """Load vocabulary file and return {mora: index} dict.

        Blank lines are skipped and IDs are assigned starting from 0,
        matching the convention of sheet's NonIntrusiveDataset._load_mora_vocab.
        (ID 0 is the first non-blank entry; the Embedding layer zeros out
        padding_idx=0 at inference time.)
        """
        vocab = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                mora = line.rstrip("\n")
                if mora == "":
                    continue
                vocab[mora] = len(vocab)
        return vocab

    def _mora_to_ids(self, mora: List[str]) -> torch.Tensor:
        ids = [self.mora_vocab.get(m, 0) for m in mora]  # 0 = unk (matches sheet)
        return torch.tensor(ids, dtype=torch.long)

    @torch.no_grad()
    def predict(
        self,
        mora: List[str],
        wav_path: Optional[Union[str, Path]] = None,
        wav: Optional[torch.Tensor] = None,
        sampling_rate: int = 16000,
    ) -> Dict:
        """Run inference.

        Args:
            mora: List of katakana mora tokens (required).
            wav_path: Path to audio file (mutually exclusive with wav).
                      Loaded via torchaudio replicating sheet's read_waveform()
                      behavior exactly (including the channels_first=False
                      resampling quirk).
            wav: Pre-loaded waveform tensor, shape (samples,) or (channels, samples).
                 Passed directly to the model without any resampling. Use this
                 only when you have already replicated sheet's loading behavior.
            sampling_rate: Unused; kept for API compatibility.

        Returns:
            dict with keys:
              "mos": float — predicted MOS score
              "frame_error_logits": np.ndarray or None — per-frame error probability
              "frame_lengths": int — number of encoder frames
        """
        if wav_path is None and wav is None:
            raise ValueError("Either wav_path or wav must be provided.")
        if wav_path is not None and wav is not None:
            raise ValueError("Provide only one of wav_path or wav, not both.")

        waveform = self._load_waveform(wav_path=wav_path, wav=wav, sampling_rate=sampling_rate)
        waveform = waveform.unsqueeze(0).to(self.device)  # (1, T)
        waveform_lengths = torch.tensor([waveform.shape[1]], dtype=torch.long, device=self.device)

        mora_ids = self._mora_to_ids(mora).unsqueeze(0).to(self.device)  # (1, L)
        mora_lengths = torch.tensor([len(mora)], dtype=torch.long, device=self.device)

        inputs = {
            "waveform": waveform,
            "waveform_lengths": waveform_lengths,
            "mora_idxs": mora_ids,
            "mora_lengths": mora_lengths,
        }

        out = self.model.mean_net_inference(inputs)

        mos = float(out["scores"][0].item())
        frame_lengths = int(out["frame_lengths"][0].item())

        frame_error_logits = None
        if "frame_error_logits" in out:
            logits = out["frame_error_logits"][0, :frame_lengths, 0]  # (T,)
            frame_error_logits = torch.sigmoid(logits).cpu().numpy()

        return {
            "mos": mos,
            "frame_error_logits": frame_error_logits,
            "frame_lengths": frame_lengths,
        }

    def _load_waveform(
        self,
        wav_path: Optional[Union[str, Path]],
        wav: Optional[torch.Tensor],
        sampling_rate: int,
    ) -> torch.Tensor:
        """Load and pre-process waveform to float32 tensor (samples,).

        Replicates sheet/datasets/non_intrusive.py read_waveform() exactly:
          1. torchaudio.load(..., channels_first=False) → (T, C)
          2. torchaudio.transforms.Resample(sr, 16000) applied to (T, C) tensor
             NOTE: Resample acts on the last dimension, so for mono (T, 1) audio
             it resamples each of the T frames as a 1-sample signal — effectively
             scaling the amplitude through the filter kernel rather than
             changing the number of time steps. This is the training-time
             behavior and must be reproduced at inference.
          3. Multi-channel → mean over channel dim
          4. squeeze to (T,)
          5. adjust_length (pad to 1040, truncate to 10 s)
        """
        import torchaudio

        TARGET_SR = 16000
        min_samples = 1040
        max_samples = TARGET_SR * 10  # 10 seconds at 16 kHz (same constant as sheet)

        if wav_path is not None:
            waveform, sr = torchaudio.load(str(wav_path), channels_first=False)  # (T, C)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR, dtype=waveform.dtype)
                waveform = resampler(waveform)  # acts on last dim (C), not T — matches sheet
            if waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1, keepdim=True)
            waveform = waveform.squeeze(-1)  # (T,)
        else:
            assert wav is not None
            waveform = wav.float()
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        # adjust_length (matches sheet)
        if waveform.shape[0] < min_samples:
            to_pad = (min_samples - waveform.shape[0]) // 2
            waveform = torch.nn.functional.pad(waveform, (to_pad, to_pad), "constant", 0)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        return waveform.float()
