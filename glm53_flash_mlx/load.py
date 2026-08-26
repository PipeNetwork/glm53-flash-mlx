"""Load GLM-5.3-Flash through this package's runtime, from a raw HF (bf16 or FP8) or converted checkpoint.

`mlx_vlm.load` resolves `mlx_vlm.models.<model_type>` from the installed package, so it cannot pick
up the fixed runtime here; this replays the same steps with our classes. Quantized checkpoints
carry a `quantization` map in config.json (uniform plus per-module overrides), replayed with
`nn.quantize` the way mlx-vlm does, so mixed-precision builds reload exactly as built.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .glm5_next import Model, ModelConfig, TextConfig, VisionConfig


def make_config(d: dict) -> ModelConfig:
    cfg = ModelConfig.from_dict(d)
    cfg.text_config = TextConfig.from_dict(d["text_config"])
    cfg.vision_config = VisionConfig.from_dict(d["vision_config"])
    return cfg


def load_model(path, lazy: bool = False, strict: bool = True, skip_vision: bool = False):
    path = Path(path)
    with open(path / "config.json") as fh:
        config = json.load(fh)
    model = Model(make_config(config))

    weights = {}
    for wf in sorted(glob.glob(str(path / "*.safetensors"))):
        weights.update(mx.load(wf))
    weights = model.sanitize(weights)

    if (quantization := config.get("quantization")) is not None:
        def class_predicate(p, m):
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    class_predicate=class_predicate)

    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model, config


def load(path, lazy: bool = False):
    from mlx_vlm.utils import load_processor
    model, config = load_model(path, lazy=lazy)
    processor = load_processor(Path(path), add_detokenizer=True)
    return model, processor
