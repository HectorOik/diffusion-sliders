# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Dummy system message for API compatibility with steering scripts
SYSTEM_MESSAGE = ""


def format_input(
    prompts: List[str],
    system_message: str = SYSTEM_MESSAGE,
    images: Optional[Union[List[PIL.Image.Image], List[List[PIL.Image.Image]]]] = None,
) -> List[str]:
    """
    Passthrough prompt formatting for FLUX.1 (FLUX.1 does not require chat templates).
    """
    return [prompt.replace("[IMG]", "").strip() for prompt in prompts]


class Flux1Pipeline(FluxPipeline):
    r"""
    The FLUX.1-dev pipeline for text-to-image generation and text-space steering.
    Inherits from Diffusers' native `FluxPipeline` (CLIP-L + T5-XXL dual text stack).
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        self.system_message = SYSTEM_MESSAGE

    @staticmethod
    def _get_module_execution_device(module: Optional[torch.nn.Module]) -> torch.device:
        """Helper to reliably locate execution device across device maps or standard setups."""
        if module is None:
            return torch.device("cpu")

        for submodule in module.modules():
            if (
                hasattr(submodule, "_hf_hook")
                and hasattr(submodule._hf_hook, "execution_device")
                and submodule._hf_hook.execution_device is not None
            ):
                return torch.device(submodule._hf_hook.execution_device)

        if hasattr(module, "hf_device_map") and module.hf_device_map:
            first_device = next(iter(module.hf_device_map.values()))
            if first_device not in ("cpu", "disk"):
                if isinstance(first_device, int):
                    return torch.device(f"cuda:{first_device}")
                return torch.device(first_device)

        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Natively encodes prompts into FLUX.1's joint text embedding space.
        """
        device = device or self._execution_device

        if prompt is None and prompt_embeds is None:
            prompt = ""

        return super().encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )


__all__ = [
    "Flux1Pipeline",
    "SYSTEM_MESSAGE",
    "format_input",
]