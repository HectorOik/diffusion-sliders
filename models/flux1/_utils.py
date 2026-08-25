"""Shared Flux 1 utilities used by inference, compute_vectors, and elastic_band.

All generic steering utilities (compute_difference_of_means, find_all_spans, etc.)
live in the top-level `steering` package. This module holds only the parts tightly
coupled to the Flux 1 pipeline and T5/CLIP text encoders.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import numpy as np
from transformers import BitsAndBytesConfig
from diffusers import FluxPipeline, FluxTransformer2DModel

from steering import DTYPE_MAP, find_all_spans, split_style_terms

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_REPO = ""  # Set if using a specific FLUX.1 LoRA
LORA_WEIGHT = ""

MAX_SEQUENCE_LENGTH = 512


# ---------------------------------------------------------------------------
# Prompt preparation and token alignment (T5-based for FLUX.1)
# ---------------------------------------------------------------------------


def prepare_flux1_text_inputs(
    tokenizer,
    prompt: str,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> dict:
    """Tokenize *prompt* via the T5 tokenizer and return offset + alignment metadata."""
    tokenized = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_sequence_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    # Also get non-padded plain tokenized representation for span matching
    plain_tokenized = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
    )

    return {
        "formatted_prompt": prompt,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "plain_offsets": plain_tokenized["offset_mapping"],
        "plain_input_ids": plain_tokenized["input_ids"],
    }


def get_style_token_positions(
    user_prompt: str,
    style: str,
    plain_offsets: Sequence[tuple[int, int]],
    plain_input_ids: Sequence[int],
    special_token_ids: Sequence[int],
) -> List[int]:
    """Return the input-sequence positions of tokens overlapping the style span in FLUX.1 text space."""
    terms = split_style_terms(style)
    if not terms:
        return []

    spans_global: List[tuple[int, int]] = [
        (start, end)
        for term in terms
        for start, end in find_all_spans(user_prompt, term)
    ]
    if not spans_global:
        return []

    special_ids = {int(tid) for tid in special_token_ids}
    matched_positions: set[int] = set()
    
    for plain_idx, (tok_start, tok_end) in enumerate(plain_offsets):
        if tok_start == tok_end:
            continue
        if not any(tok_start < span_end and tok_end > span_start for span_start, span_end in spans_global):
            continue
        token_id = int(plain_input_ids[plain_idx])
        if token_id in special_ids:
            continue
        # T5 token position matches index directly with offset adjustments
        matched_positions.add(plain_idx)

    return sorted(matched_positions)


# ---------------------------------------------------------------------------
# Token-index lookup (used by both inference and elastic_band)
# ---------------------------------------------------------------------------


def find_indices_to_edit(
    pipe: FluxPipeline,
    prompt: str,
    tokens_to_edit: Sequence[str],
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> List[int]:
    """Return prompt-embedding positions for *tokens_to_edit* within *prompt* using T5 tokenizer."""
    tokenizer = pipe.tokenizer_2 if hasattr(pipe, "tokenizer_2") else pipe.tokenizer
    
    try:
        prepared = prepare_flux1_text_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )
        style = " ".join(
            part
            for token in tokens_to_edit
            for part in split_style_terms(token)
        )
        positions = get_style_token_positions(
            user_prompt=prompt,
            style=style,
            plain_offsets=prepared["plain_offsets"],
            plain_input_ids=prepared["plain_input_ids"],
            special_token_ids=getattr(tokenizer, "all_special_ids", []),
        )
    except Exception:
        positions = []

    if not positions:
        # Fallback to middle sequence content tokens if exact string spans fail
        max_len = min(max_sequence_length - 1, 15)
        positions = list(range(1, max_len))
        
    return sorted([int(p) for p in positions if p < max_sequence_length])


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------


def apply_steering(prompt_embeds, idx_to_edit, steering_vec, factor):
    # Ensure steering_vec is a 1D tensor of shape [12288] to match prompt_embeds[batch_idx, idx, :]
    if isinstance(steering_vec, np.ndarray):
        steering_vec = torch.tensor(steering_vec, dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    steering_vec = steering_vec.squeeze() # Removes any leading batch dimensions like [1, 12288] -> [12288]

    if steering_vec.shape[-1] != prompt_embeds.shape[-1]:
        raise ValueError(f"Steering vector width {steering_vec.shape[-1]} does not match prompt embeddings {prompt_embeds.shape[-1]}.")

    # Forcefully escape inference mode restrictions by using .data.clone()
    with torch.enable_grad():
        prompt_embeds = prompt_embeds.detach().data.clone()

        for batch_idx in range(prompt_embeds.shape[0]):
            for idx in idx_to_edit:
                slice_to_steer = prompt_embeds[batch_idx, idx, :]
                prompt_embeds[batch_idx, idx, :] = slice_to_steer + (factor * steering_vec)
            
    return prompt_embeds


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------


def pool_positions(sequence_tensor: torch.Tensor, positions: Sequence[int]) -> Optional[torch.Tensor]:
    """Mean-pool *sequence_tensor[0]* at *positions* — strips the leading batch dim."""
    if not positions:
        return None
    return torch.stack(
        [sequence_tensor[0, idx].to(dtype=torch.float32).cpu() for idx in positions]
    ).mean(0)


def build_pipeline(
    torch_dtype: torch.dtype,
    use_lora: bool,
    use_distributed: bool,
    pipeline_device: str = "cuda",
) -> FluxPipeline:
    print("⚡ Loading 4-bit NF4 Quantized FLUX.1-dev Transformer...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
    )
    
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch_dtype,
    ).to(pipeline_device)

    if use_lora and LORA_REPO:
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT)
    return pipe
