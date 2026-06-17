#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SSLMOS model — ported from sheet/models/sslmos.py
# Imports changed from sheet.* to pasqa._sheet.*

import math
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from pasqa._sheet.grl import grad_reverse
from pasqa._sheet.modules import MoraCrossAttention, Projection, make_non_pad_mask


def _patch_torchaudio_for_s3prl():
    """Patch torchaudio to restore APIs removed in newer versions, for s3prl compat."""
    import torchaudio as _ta

    # set_audio_backend removed in torchaudio>=2.5
    if not hasattr(_ta, "set_audio_backend"):
        _ta.set_audio_backend = lambda *a, **kw: None

    # torchaudio.sox_effects removed in torchaudio>=2.5
    if "torchaudio.sox_effects" not in sys.modules:
        sox_stub = types.ModuleType("torchaudio.sox_effects")

        def _apply_effects_tensor(waveform, sample_rate, effects, channels_first=True):
            # Minimal stub: return audio unchanged
            return waveform, sample_rate

        sox_stub.apply_effects_tensor = _apply_effects_tensor
        sox_stub.apply_effects_file = None
        sys.modules["torchaudio.sox_effects"] = sox_stub
        _ta.sox_effects = sox_stub


class SSLMOS(torch.nn.Module):
    def __init__(
        self,
        # dummy, for signature need
        model_input: str,
        # model related
        ssl_module: str = "s3prl",
        s3prl_name: str = "wav2vec2",
        ssl_model_output_dim: int = 768,
        ssl_model_layer_idx: int = -1,
        # mean net related
        mean_net_dnn_dim: int = 64,
        mean_net_output_type: str = "scalar",
        mean_net_output_dim: int = 5,
        mean_net_output_step: float = 0.25,
        mean_net_range_clipping: bool = True,
        # frame error head
        use_error_head: bool = False,
        error_head_dnn_dim: int = 64,
        # error-conditioned MOS
        use_error_conditioned_mos: bool = False,
        error_deduction_lambda: float = 1.0,
        error_prob_detach: bool = True,
        mos_target: str = "base",
        # mora cross-attention related
        use_mora: bool = False,
        mora_vocab_size: int = None,
        mora_emb_dim: int = 256,
        mora_transformer_layers: int = 1,
        mora_transformer_heads: int = 4,
        mora_ffn_dim: int = 512,
        mora_dropout: float = 0.1,
        mora_max_len: int = 128,
        mora_pos_encoding: str = "rope",
        attn_dim: int = 256,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        attn_alpha_init: float = 0.1,
        # speaker grl related
        use_speaker_grl: bool = False,
        num_speakers: int = None,
        speaker_grl_hidden_dim: int = 128,
        speaker_grl_dropout: float = 0.1,
        # listener related (kept for signature compatibility)
        use_listener_modeling: bool = False,
        num_listeners: int = None,
        listener_emb_dim: int = None,
        use_mean_listener: bool = True,
        # decoder related
        decoder_type: str = "ffn",
        decoder_dnn_dim: int = 64,
        output_type: str = "scalar",
        range_clipping: bool = True,
        # additional head (for RAMP)
        use_additional_categorical_head: bool = False,
        categorical_head_dnn_dim: int = 64,
        categorical_head_output_dim: int = 17,
        categorical_head_output_step: float = 0.25,
        categorical_head_range_clipping: bool = True,
        # dummy, for signature need
        num_domains: int = None,
        # world frontend (unused in inference, kept for signature compatibility)
        world_frame_time: float = 0.01,
        world_f0_min: int = 50,
        world_f0_max: int = 500,
        world_log_f0: bool = True,
        world_sampling_rate: int = 16000,
        world_use_mcep: bool = False,
        world_mcep_dim: int = 24,
        world_use_ap: bool = False,
        world_ap_dim: int = 1,
    ):
        super().__init__()
        self.use_mean_listener = use_mean_listener
        self.output_type = output_type
        self.use_additional_categorical_head = use_additional_categorical_head
        self.use_listener_modeling = use_listener_modeling

        # define ssl model
        self.ssl_module = ssl_module
        if ssl_module == "s3prl":
            _patch_torchaudio_for_s3prl()
            from s3prl.nn import S3PRLUpstream

            if s3prl_name in S3PRLUpstream.available_names():
                self.ssl_model = S3PRLUpstream(s3prl_name)
            self.ssl_model_layer_idx = ssl_model_layer_idx
        else:
            raise NotImplementedError(f"ssl_module '{ssl_module}' not supported")

        # mora cross-attention
        self.use_mora = use_mora
        if self.use_mora:
            if mora_vocab_size is None:
                raise ValueError("mora_vocab_size must be provided when use_mora=True.")
            self.mora_attn = MoraCrossAttention(
                encoder_dim=ssl_model_output_dim,
                mora_vocab_size=mora_vocab_size,
                mora_emb_dim=mora_emb_dim,
                mora_transformer_layers=mora_transformer_layers,
                mora_transformer_heads=mora_transformer_heads,
                mora_ffn_dim=mora_ffn_dim,
                mora_dropout=mora_dropout,
                mora_max_len=mora_max_len,
                mora_pos_encoding=mora_pos_encoding,
                attn_dim=attn_dim,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
                attn_alpha_init=attn_alpha_init,
            )

        # mean net
        self.mean_net_dnn = Projection(
            ssl_model_output_dim,
            mean_net_dnn_dim,
            nn.ReLU,
            mean_net_output_type,
            mean_net_output_dim,
            mean_net_output_step,
            mean_net_range_clipping,
        )
        self.mean_net_range_clipping = mean_net_range_clipping

        # frame error head
        self.use_error_head = use_error_head
        if self.use_error_head:
            self.error_head = Projection(
                ssl_model_output_dim,
                error_head_dnn_dim,
                nn.ReLU,
                "scalar",
                1,
                range_clipping=False,
            )

        # error-conditioned MOS
        self.use_error_conditioned_mos = use_error_conditioned_mos
        self.error_deduction_lambda = float(error_deduction_lambda)
        self.error_prob_detach = error_prob_detach
        self.mos_target = mos_target

        # speaker GRL
        self.use_speaker_grl = use_speaker_grl
        if self.use_speaker_grl:
            if num_speakers is None:
                raise ValueError(
                    "num_speakers must be provided when use_speaker_grl=True."
                )
            self.speaker_classifier = nn.Sequential(
                nn.Linear(ssl_model_output_dim, speaker_grl_hidden_dim),
                nn.ReLU(),
                nn.Dropout(speaker_grl_dropout),
                nn.Linear(speaker_grl_hidden_dim, num_speakers),
            )

        # additional categorical head
        if use_additional_categorical_head:
            self.categorical_head = Projection(
                ssl_model_output_dim,
                mean_net_dnn_dim,
                nn.ReLU,
                "categorical",
                categorical_head_output_dim,
                categorical_head_output_step,
                categorical_head_range_clipping,
            )

    def get_num_params(self):
        return sum(p.numel() for n, p in self.named_parameters())

    def _apply_mora_attention(self, encoder_outputs, inputs):
        if not self.use_mora:
            return encoder_outputs
        if "mora_idxs" not in inputs:
            raise ValueError("mora_idxs must be provided when use_mora=True.")
        return self.mora_attn(
            encoder_outputs,
            inputs["mora_idxs"],
            inputs.get("mora_lengths", None),
        )

    @staticmethod
    def _masked_mean(x, lengths):
        mask = make_non_pad_mask(lengths).to(x.device)
        mask = mask.unsqueeze(-1).to(x.dtype)
        lengths = lengths.to(x.device).unsqueeze(-1).to(x.dtype).clamp_min(1)
        return (x * mask).sum(dim=1) / lengths

    def _apply_error_conditioning(self, mean_scores, error_logits):
        error_probs = torch.sigmoid(error_logits)
        if self.error_prob_detach:
            error_probs = error_probs.detach()
        adjusted = mean_scores - self.error_deduction_lambda * error_probs
        if self.mean_net_range_clipping:
            adjusted = torch.clamp(adjusted, min=1.0, max=5.0)
        return adjusted

    def mean_net_inference(self, inputs):
        waveform = inputs["waveform"]
        waveform_lengths = inputs["waveform_lengths"]

        all_encoder_outputs, all_encoder_outputs_lens = self.ssl_model(
            waveform, waveform_lengths
        )
        encoder_outputs = all_encoder_outputs[self.ssl_model_layer_idx]
        encoder_outputs_lens = all_encoder_outputs_lens[self.ssl_model_layer_idx]
        encoder_outputs = self._apply_mora_attention(encoder_outputs, inputs)

        decoder_inputs = encoder_outputs
        mean_net_outputs = self.mean_net_dnn(
            decoder_inputs, inference=True
        )
        mean_scores_base = mean_net_outputs.to(torch.float)
        scores_base = self._masked_mean(mean_scores_base, encoder_outputs_lens).squeeze(-1)
        scores = scores_base
        error_logits = None
        if self.use_error_head:
            error_logits = self.error_head(encoder_outputs)
        if self.use_error_conditioned_mos:
            mean_scores_adjusted = self._apply_error_conditioning(
                mean_scores_base, error_logits
            )
            scores_adjusted = self._masked_mean(
                mean_scores_adjusted, encoder_outputs_lens
            ).squeeze(-1)
            scores = scores_adjusted if self.mos_target == "adjusted" else scores_base

        ret = {
            "ssl_embeddings": encoder_outputs,
            "scores": scores,
            "frame_lengths": encoder_outputs_lens,
        }
        if self.use_error_head and error_logits is not None:
            ret["frame_error_logits"] = error_logits
        if self.use_additional_categorical_head:
            ret["confidences"] = self.categorical_head(decoder_inputs)

        return ret
