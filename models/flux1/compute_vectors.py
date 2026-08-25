#!/usr/bin/env python3
"""Compute a Flux 1 steering vector from T5 prompt embeddings.

Usage
-----
python -m models.flux1.compute_vectors \\
    --pairs_file path/to/cartoon.jsonl \\
    --out_dir models/flux1/assets/my_concept
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from steering import (
    DTYPE_MAP,
    compute_difference_of_means,
    save_steering_outputs,
    validate_max_pairs,
    validate_path_exists,
)
from ._utils import (
    MAX_SEQUENCE_LENGTH,
    build_pipeline,
    find_indices_to_edit,
    pool_positions,
)

TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)


# ---------------------------------------------------------------------------
# Style representation collection for FLUX.1 (T5 encoder)
# ---------------------------------------------------------------------------


def collect_style_representations(args: argparse.Namespace) -> tuple[List[np.ndarray], List[int]]:
    dtype = DTYPE_MAP[args.dtype]
    if args.device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    # Load FLUX.1 pipeline to access text_encoder_2 (T5-XXL)
    pipe = build_pipeline(torch_dtype=dtype, use_lora=False, use_distributed=False, pipeline_device=args.device)
    t5_encoder = pipe.text_encoder_2 if hasattr(pipe, "text_encoder_2") else pipe.text_encoder
    t5_tokenizer = pipe.tokenizer_2 if hasattr(pipe, "tokenizer_2") else pipe.tokenizer

    style_vectors: List[np.ndarray] = []
    style_labels: List[int] = []
    processed = 0

    with args.pairs_file.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Collecting Flux 1 prompt spans"):
            if args.max_pairs != -1 and processed >= args.max_pairs:
                break

            example = json.loads(line)
            pos_text, neg_text = example["pos"], example["neg"]
            pos_style, neg_style = example.get("pos_style"), example.get("neg_style")
            if not pos_style or not neg_style:
                raise ValueError(
                    f"Each example must include pos_style and neg_style; "
                    f"got pos_style={pos_style}, neg_style={neg_style}"
                )

            # Find token indices to target for both prompts
            pos_positions = find_indices_to_edit(pipe, pos_text, [pos_style], args.max_sequence_length)
            neg_positions = find_indices_to_edit(pipe, neg_text, [neg_style], args.max_sequence_length)

            if not pos_positions or not neg_positions:
                raise ValueError(
                    f"Failed to align style spans to Flux 1 T5 text tokens. "
                    f"pos_style={pos_style}, neg_style={neg_style}"
                )

            # Tokenize and run through T5 encoder directly to get multi-layer hidden states
            with torch.inference_mode():
                pos_tokens = t5_tokenizer(
                    pos_text, return_tensors="pt", padding="max_length", 
                    truncation=True, max_length=args.max_sequence_length
                ).to(args.device)
                
                neg_tokens = t5_tokenizer(
                    neg_text, return_tensors="pt", padding="max_length", 
                    truncation=True, max_length=args.max_sequence_length
                ).to(args.device)

                pos_out = t5_encoder(input_ids=pos_tokens.input_ids, output_hidden_states=True, use_cache=False)
                neg_out = t5_encoder(input_ids=neg_tokens.input_ids, output_hidden_states=True, use_cache=False)

                # Extract and concatenate selected layers
                pos_hidden = torch.stack([pos_out.hidden_states[i] for i in args.text_encoder_out_layers], dim=1)
                neg_hidden = torch.stack([neg_out.hidden_states[i] for i in args.text_encoder_out_layers], dim=1)

                # Reshape to match multi-layer hidden embedding structure
                b, l, s, h = pos_hidden.shape
                pos_embeds = pos_hidden.permute(0, 2, 1, 3).reshape(b, s, l * h)
                neg_embeds = neg_hidden.permute(0, 2, 1, 3).reshape(b, s, l * h)

            pos_vector = pool_positions(pos_embeds, pos_positions)
            neg_vector = pool_positions(neg_embeds, neg_positions)
            if pos_vector is None or neg_vector is None:
                raise ValueError(
                    f"Failed to pool Flux 1 embeddings for style spans. "
                    f"pos_style={pos_style}, neg_style={neg_style}"
                )

            style_vectors.append(pos_vector.numpy().astype(np.float32))
            style_labels.append(1)
            style_vectors.append(neg_vector.numpy().astype(np.float32))
            style_labels.append(0)
            processed += 1

    if not style_vectors:
        raise ValueError("No valid style vectors were collected from the provided pairs file.")
    return style_vectors, style_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Flux 1 steering vectors from contrastive prompt pairs."
    )
    parser.add_argument("--pairs_file", type=validate_path_exists, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--max_pairs", type=validate_max_pairs, default=-1)
    parser.add_argument("--max_sequence_length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=list(TEXT_ENCODER_OUT_LAYERS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vectors, labels = collect_style_representations(args)
    steering, max_projection, min_projection = compute_difference_of_means(vectors, labels)
    save_steering_outputs(args.out_dir, steering, max_projection, min_projection)


if __name__ == "__main__":
    main()