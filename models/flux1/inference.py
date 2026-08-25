#!/usr/bin/env python3
"""Flux 1 steering inference — generates a grid across a range of strengths.

Usage
-----
python -m models.flux1.inference \\
    --input_image path/to/image.png \\
    --steering_vector models/flux1/assets/cartoon/steering_last_layer.npy \\
    --prompt "make the scene cartoon" \\
    --tokens_to_edit cartoon \\
    --strengths -2.0 -1.0 0.0 1.0 2.0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import load_image
from PIL import Image

from steering import DTYPE_MAP, load_steering_vector
from ._utils import (
    MAX_SEQUENCE_LENGTH,
    apply_steering,
    build_pipeline,
    find_indices_to_edit,
)

TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def make_grid(images: list[Image.Image], cols: int = 4, pad: int = 16) -> Image.Image:
    if not images:
        raise ValueError("No images provided for grid creation.")
    w, h = images[0].size
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        grid.paste(image, (col * (w + pad), row * (h + pad)))
    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flux 1 steering inference.")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--steering_vector", type=str, required=True,
                        help="Path to steering_last_layer.npy")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tokens_to_edit", type=str, nargs="+", required=True)
    parser.add_argument("--strengths", type=float, nargs="+", required=True,
                        help="Steering strengths to sweep (e.g. -2.0 -1.0 0.0 1.0 2.0)")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_flux1_steering"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--lora", action="store_true", default=False,
                        help="Load LoRA weights if configured.")
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+",
                        default=list(TEXT_ENCODER_OUT_LAYERS))
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    guidance_scale = args.guidance_scale
    num_inference_steps = 28

    pipe = build_pipeline(torch_dtype=DTYPE_MAP[args.dtype], use_lora=args.lora, use_distributed=args.distributed)
    condition_image = load_image(args.input_image).convert("RGB")
    steering_vector = load_steering_vector(args.steering_vector, device="cpu")

    idx_to_edit = find_indices_to_edit(
        pipe=pipe,
        prompt=args.prompt,
        tokens_to_edit=args.tokens_to_edit,
        max_sequence_length=args.max_sequence_length,
    )
    print(f"Editing token indices: {idx_to_edit}")

    t5_encoder = getattr(pipe, "text_encoder_2", pipe.text_encoder)
    t5_tokenizer = getattr(pipe, "tokenizer_2", pipe.tokenizer)
    text_encoder_device = next(t5_encoder.parameters()).device

    # Encode base prompt multi-layer hidden states via T5-XXL
    tokens = t5_tokenizer(
        args.prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=args.max_sequence_length
    ).to(text_encoder_device)

    out = t5_encoder(input_ids=tokens.input_ids, output_hidden_states=True, use_cache=False)
    hidden = torch.stack([out.hidden_states[i] for i in args.text_encoder_out_layers], dim=1)
    b, l, s, h = hidden.shape
    base_prompt_embeds = hidden.permute(0, 2, 1, 3).reshape(b, s, l * h).to(dtype=pipe.dtype)

    # Obtain pooled prompt embeddings natively
    _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, prompt_2=None, device=text_encoder_device, max_sequence_length=args.max_sequence_length
    )

    generated_images: list[Image.Image] = []
    for strength in args.strengths:
        prompt_embeds = apply_steering(
            base_prompt_embeds=base_prompt_embeds,
            idx_to_edit=idx_to_edit,
            steering_vector=steering_vector,
            factor=strength,
        )
        output = pipe(
            image=condition_image,
            prompt=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
        image = output.images[0]
        out_path = args.out_dir / f"steer_{strength:.3f}.png"
        image.save(out_path)
        generated_images.append(image)
        print(f"Saved: {out_path}")

    grid = make_grid(generated_images, cols=min(4, len(generated_images)))
    grid_path = args.out_dir / "grid.png"
    grid.save(grid_path)
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()