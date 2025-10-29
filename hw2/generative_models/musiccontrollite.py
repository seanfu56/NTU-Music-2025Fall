# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from safetensors.torch import load_file
from diffusers.loaders import AttnProcsLayers
# from utils.extract_conditions import compute_melody, compute_melody_v2, compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
from madmom.audio.signal import Signal
import numpy as np
import matplotlib.pyplot as plt
import os
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import soundfile as sf
import random
import math
import inspect

from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from diffusers.models import AutoencoderOobleck, StableAudioDiTModel
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.schedulers import EDMDPMSolverMultistepScheduler
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
from torch.cuda.amp import autocast, GradScaler


def _patch_stable_audio_with_conditions():
    if getattr(StableAudioDiTModel, "_musecontrollite_patched", False):
        return

    def _ditblock_forward_with_conditions(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states_con=encoder_hidden_states_con,
            encoder_hidden_states_audio=encoder_hidden_states_audio,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states

    def _ditmodel_forward_with_conditions(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_hidden_states_con: Optional[torch.FloatTensor] = None,
        encoder_hidden_states_audio: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Modified StableAudioDiTModel forward method to support additional conditioning inputs.
        """
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        cross_attention_hidden_states_con = (
            self.cross_attention_proj(encoder_hidden_states_con)
            if encoder_hidden_states_con is not None
            else None
        )
        cross_attention_hidden_states_audio = (
            self.cross_attention_proj(encoder_hidden_states_audio)
            if encoder_hidden_states_audio is not None
            else None
        )
        global_hidden_states = self.global_proj(global_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.proj_in(hidden_states)

        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                    cross_attention_hidden_states_con,
                    cross_attention_hidden_states_audio,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cross_attention_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                    encoder_hidden_states_con=cross_attention_hidden_states_con,
                    encoder_hidden_states_audio=cross_attention_hidden_states_audio,
                )

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    StableAudioDiTBlock.forward = _ditblock_forward_with_conditions
    StableAudioDiTModel.forward = _ditmodel_forward_with_conditions
    StableAudioDiTModel._musecontrollite_patched = True


_patch_stable_audio_with_conditions()

def check_and_print_non_float32_parameters(model):
    non_float32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append((name, param.dtype))
    
    if non_float32_params:
        print("Not all parameters are in float32!")
        print("The following parameters are not in float32:")
        for name, dtype in non_float32_params:
            print(f"Parameter: {name}, Data Type: {dtype}")
    else:
        print("All parameters are in float32.")
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> import soundfile as sf
        >>> from diffusers import StableAudioPipeline

        >>> repo_id = "stabilityai/stable-audio-open-1.0"
        >>> pipe = StableAudioPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_end_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> output = audio[0].T.float().cpu().numpy()
        >>> sf.write("hammer.wav", output, pipe.vae.sampling_rate)
        ```
"""


class StableAudioPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using StableAudio.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderOobleck`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.T5EncoderModel`]):
            Frozen text-encoder. StableAudio uses the encoder of
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) variant.
        projection_model ([`StableAudioProjectionModel`]):
            A trained model used to linearly project the hidden-states from the text encoder model and the start and
            end seconds. The projected hidden-states from the encoder and the conditional seconds are concatenated to
            give the input to the transformer model.
        tokenizer ([`~transformers.T5Tokenizer`]):
            Tokenizer to tokenize text for the frozen text-encoder.
        transformer ([`StableAudioDiTModel`]):
            A `StableAudioDiTModel` to denoise the encoded audio latents.
        scheduler ([`EDMDPMSolverMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded audio latents.
    """

    model_cpu_offload_seq = "text_encoder->projection_model->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderOobleck,
        text_encoder: T5EncoderModel,
        projection_model: StableAudioProjectionModel,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
        transformer: StableAudioDiTModel,
        scheduler: EDMDPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            projection_model=projection_model,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # 1. Tokenize text
            self.tokenizer.model_max_length = 512
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    f"The following part of your input was truncated because {self.text_encoder.config.model_type} can "
                    f"only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 2. Text encoder forward
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if do_classifier_free_guidance and negative_prompt is not None:
            uncond_tokens: List[str]
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # 1. Tokenize text
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            # 2. Text encoder forward
            self.text_encoder.eval()
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=negative_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            if negative_attention_mask is not None:
                # set the masked tokens to the null embed
                negative_prompt_embeds = torch.where(
                    negative_attention_mask.to(torch.bool).unsqueeze(2), negative_prompt_embeds, 0.0
                )

        # 3. Project prompt_embeds and negative_prompt_embeds
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the negative and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds, prompt_embeds])
            if attention_mask is not None and negative_attention_mask is None:
                negative_attention_mask = torch.ones_like(attention_mask)
            elif attention_mask is None and negative_attention_mask is not None:
                attention_mask = torch.ones_like(negative_attention_mask)
            if attention_mask is not None:
                attention_mask = torch.cat([negative_attention_mask, attention_mask, attention_mask, attention_mask])

        prompt_embeds = self.projection_model(
            text_hidden_states=prompt_embeds,
        ).text_hidden_states
        if attention_mask is not None:
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

        return prompt_embeds

    def encode_duration(
        self,
        audio_start_in_s,
        audio_end_in_s,
        device,
        do_classifier_free_guidance,
        batch_size,
    ):
        audio_start_in_s = audio_start_in_s if isinstance(audio_start_in_s, list) else [audio_start_in_s]
        audio_end_in_s = audio_end_in_s if isinstance(audio_end_in_s, list) else [audio_end_in_s]

        if len(audio_start_in_s) == 1:
            audio_start_in_s = audio_start_in_s * batch_size
        if len(audio_end_in_s) == 1:
            audio_end_in_s = audio_end_in_s * batch_size

        # Cast the inputs to floats
        audio_start_in_s = [float(x) for x in audio_start_in_s]
        audio_start_in_s = torch.tensor(audio_start_in_s).to(device)

        audio_end_in_s = [float(x) for x in audio_end_in_s]
        audio_end_in_s = torch.tensor(audio_end_in_s).to(device)

        projection_output = self.projection_model(
            start_seconds=audio_start_in_s,
            end_seconds=audio_end_in_s,
        )
        seconds_start_hidden_states = projection_output.seconds_start_hidden_states
        seconds_end_hidden_states = projection_output.seconds_end_hidden_states

        # For classifier free guidance, we need to do two forward passes.
        # Here we repeat the audio hidden states to avoid doing two forward passes
        if do_classifier_free_guidance:
            seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states, seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
            seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states, seconds_end_hidden_states, seconds_end_hidden_states], dim=0)

        return seconds_start_hidden_states, seconds_end_hidden_states

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        audio_start_in_s,
        audio_end_in_s,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        attention_mask=None,
        negative_attention_mask=None,
        initial_audio_waveforms=None,
        initial_audio_sampling_rate=None,
    ):
        if audio_end_in_s < audio_start_in_s:
            raise ValueError(
                f"`audio_end_in_s={audio_end_in_s}' must be higher than 'audio_start_in_s={audio_start_in_s}` but "
            )

        if (
            audio_start_in_s < self.projection_model.config.min_value
            or audio_start_in_s > self.projection_model.config.max_value
        ):
            raise ValueError(
                f"`audio_start_in_s` must be greater than or equal to {self.projection_model.config.min_value}, and lower than or equal to {self.projection_model.config.max_value} but "
                f"is {audio_start_in_s}."
            )

        if (
            audio_end_in_s < self.projection_model.config.min_value
            or audio_end_in_s > self.projection_model.config.max_value
        ):
            raise ValueError(
                f"`audio_end_in_s` must be greater than or equal to {self.projection_model.config.min_value}, and lower than or equal to {self.projection_model.config.max_value} but "
                f"is {audio_end_in_s}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and (prompt_embeds is None):
            raise ValueError(
                "Provide either `prompt`, or `prompt_embeds`. Cannot leave"
                "`prompt` undefined without specifying `prompt_embeds`."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if attention_mask is not None and attention_mask.shape != prompt_embeds.shape[:2]:
                raise ValueError(
                    "`attention_mask should have the same batch size and sequence length as `prompt_embeds`, but got:"
                    f"`attention_mask: {attention_mask.shape} != `prompt_embeds` {prompt_embeds.shape}"
                )

        if initial_audio_sampling_rate is None and initial_audio_waveforms is not None:
            raise ValueError(
                "`initial_audio_waveforms' is provided but the sampling rate is not. Make sure to pass `initial_audio_sampling_rate`."
            )

        if initial_audio_sampling_rate is not None and initial_audio_sampling_rate != self.vae.sampling_rate:
            raise ValueError(
                f"`initial_audio_sampling_rate` must be {self.vae.hop_length}' but is `{initial_audio_sampling_rate}`."
                "Make sure to resample the `initial_audio_waveforms` and to correct the sampling rate. "
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_vae,
        sample_size,
        dtype,
        device,
        generator,
        latents=None,
        initial_audio_waveforms=None,
        num_waveforms_per_prompt=None,
        audio_channels=None,
    ):
        shape = (batch_size, num_channels_vae, sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # encode the initial audio for use by the model
        if initial_audio_waveforms is not None:
            # check dimension
            if initial_audio_waveforms.ndim == 2:
                initial_audio_waveforms = initial_audio_waveforms.unsqueeze(1)
            elif initial_audio_waveforms.ndim != 3:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but has `{initial_audio_waveforms.ndim}` dimensions"
                )

            audio_vae_length = self.transformer.config.sample_size * self.vae.hop_length
            audio_shape = (batch_size // num_waveforms_per_prompt, audio_channels, audio_vae_length)

            # check num_channels
            if initial_audio_waveforms.shape[1] == 1 and audio_channels == 2:
                initial_audio_waveforms = initial_audio_waveforms.repeat(1, 2, 1)
            elif initial_audio_waveforms.shape[1] == 2 and audio_channels == 1:
                initial_audio_waveforms = initial_audio_waveforms.mean(1, keepdim=True)

            if initial_audio_waveforms.shape[:2] != audio_shape[:2]:
                raise ValueError(
                    f"`initial_audio_waveforms` must be of shape `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)` but is of shape `{initial_audio_waveforms.shape}`"
                )

            # crop or pad
            audio_length = initial_audio_waveforms.shape[-1]
            if audio_length < audio_vae_length:
                logger.warning(
                    f"The provided input waveform is shorter ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be padded."
                )
            elif audio_length > audio_vae_length:
                logger.warning(
                    f"The provided input waveform is longer ({audio_length}) than the required audio length ({audio_vae_length}) of the model and will thus be cropped."
                )

            audio = initial_audio_waveforms.new_zeros(audio_shape)
            audio[:, :, : min(audio_length, audio_vae_length)] = initial_audio_waveforms[:, :, :audio_vae_length]

            encoded_audio = self.vae.encode(audio).latent_dist.sample(generator)
            encoded_audio = encoded_audio.repeat((num_waveforms_per_prompt, 1, 1))
            latents = encoded_audio + latents
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        extracted_condition_audio = None,
        extracted_condition = None,
        prompt: Union[str, List[str]] = None,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
        num_inference_steps: int = 100,
        guidance_scale_text: float = 7.0,
        guidance_scale_con: float = 2.0,
        guidance_scale_audio: float = 2.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        initial_audio_waveforms: Optional[torch.Tensor] = None,
        initial_audio_sampling_rate: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        negative_attention_mask: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        output_type: Optional[str] = "pt",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
            audio_end_in_s (`float`, *optional*, defaults to 47.55):
                Audio end index in seconds.
            audio_start_in_s (`float`, *optional*, defaults to 0):
                Audio start index in seconds.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                A higher guidance scale value encourages the model to generate audio that is closely linked to the text
                `prompt` at the expense of lower sound quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_waveforms_per_prompt (`int`, *optional*, defaults to 1):
                The number of waveforms to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            initial_audio_waveforms (`torch.Tensor`, *optional*):
                Optional initial audio waveforms to use as the initial audio waveform for generation. Must be of shape
                `(batch_size, num_channels, audio_length)` or `(batch_size, audio_length)`, where `batch_size`
                corresponds to the number of prompts passed to the model.
            initial_audio_sampling_rate (`int`, *optional*):
                Sampling rate of the `initial_audio_waveforms`, if they are provided. Must be the same as the model.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings from the text encoder model. Can be used to easily tweak text inputs,
                *e.g.* prompt weighting. If not provided, text embeddings will be computed from `prompt` input
                argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed negative text embeddings from the text encoder model. Can be used to easily tweak text
                inputs, *e.g.* prompt weighting. If not provided, negative_prompt_embeds will be computed from
                `negative_prompt` input argument.
            attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `prompt_embeds`. If not provided, attention mask will
                be computed from `prompt` input argument.
            negative_attention_mask (`torch.LongTensor`, *optional*):
                Pre-computed attention mask to be applied to the `negative_text_audio_duration_embeds`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            output_type (`str`, *optional*, defaults to `"pt"`):
                The output format of the generated audio. Choose between `"np"` to return a NumPy `np.ndarray` or
                `"pt"` to return a PyTorch `torch.Tensor` object. Set to `"latent"` to return the latent diffusion
                model (LDM) output.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated audio.
        """
        # 0. Convert audio input length from seconds to latent length
        downsample_ratio = self.vae.hop_length

        max_audio_length_in_s = self.transformer.config.sample_size * downsample_ratio / self.vae.config.sampling_rate
        if audio_end_in_s is None:
            audio_end_in_s = max_audio_length_in_s

        if audio_end_in_s - audio_start_in_s > max_audio_length_in_s:
            raise ValueError(
                f"The total audio length requested ({audio_end_in_s-audio_start_in_s}s) is longer than the model maximum possible length ({max_audio_length_in_s}). Make sure that 'audio_end_in_s-audio_start_in_s<={max_audio_length_in_s}'."
            )

        waveform_start = int(audio_start_in_s * self.vae.config.sampling_rate)
        waveform_end = int(audio_end_in_s * self.vae.config.sampling_rate)
        waveform_length = int(self.transformer.config.sample_size) #  * audio_end_in_s / 47.554
        # waveform_length = 646
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            audio_start_in_s,
            audio_end_in_s,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
            initial_audio_waveforms,
            initial_audio_sampling_rate,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True

        # 3. Encode input prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

        # Encode duration
        seconds_start_hidden_states, seconds_end_hidden_states = self.encode_duration(
            audio_start_in_s,
            audio_end_in_s,
            device,
            do_classifier_free_guidance and (negative_prompt is not None or negative_prompt_embeds is not None),
            batch_size,
        )

        # Create text_audio_duration_embeds and audio_duration_embeds
        text_audio_duration_embeds = torch.cat(
            [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
        )

        audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        # In case of classifier free guidance without negative prompt, we need to create unconditional embeddings and
        # to concatenate it to the embeddings
        if do_classifier_free_guidance and negative_prompt_embeds is None and negative_prompt is None:
            negative_text_audio_duration_embeds = torch.zeros_like(
                text_audio_duration_embeds, device=text_audio_duration_embeds.device
            )
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
            )
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)
        # if condition is not None:
        #     condition_conditioning = condition_model(condition)
        #     condition_no_conditioning = condition_model(torch.full_like(condition, fill_value=0))
        #     extracted_condition = torch.cat([condition_no_conditioning, condition_no_conditioning, condition_conditioning], dim=0)
          
        bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
        # duplicate audio_duration_embeds and text_audio_duration_embeds for each generation per prompt, using mps friendly method
        text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        text_audio_duration_embeds = text_audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        audio_duration_embeds = audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_vae = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_vae,
            waveform_length,
            text_audio_duration_embeds.dtype,
            device,
            generator,
            latents,
            initial_audio_waveforms,
            num_waveforms_per_prompt,
            audio_channels=self.vae.config.audio_channels,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare rotary positional embedding
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                with autocast():
                    noise_pred = self.transformer(
                        latent_model_input,
                        t.unsqueeze(0),
                        encoder_hidden_states=text_audio_duration_embeds,
                        encoder_hidden_states_con=extracted_condition,
                        encoder_hidden_states_audio = extracted_condition_audio,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                # transformer_weight_dtype = next(self.transformer.parameters()).dtype
                # print("transformer_weight_dtype",transformer_weight_dtype)
                # noise_pred = noise_pred.half()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_both, noise_pred_both_audio = noise_pred.chunk(4)
                    noise_pred = noise_pred_uncond + guidance_scale_text * (noise_pred_text - noise_pred_uncond) + guidance_scale_con * (noise_pred_both - noise_pred_text) \
                    + guidance_scale_audio * (noise_pred_both_audio - noise_pred_both)
                # print("guidance_scale_audio", guidance_scale_audio)
                # if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text, noise_pred_both, noise_pred_both_audio = noise_pred.chunk(4)
                #     noise_pred_uncond_no_mask, noise_pred_text_no_mask, noise_pred_both_no_mask, noise_pred_both_audio_no_mask = noise_pred_uncond[:,:,:323], noise_pred_text[:,:,:323], noise_pred_both[:,:,:323], noise_pred_both_audio[:,:,:323]
                #     noise_pred_no_mask = noise_pred_uncond_no_mask + 7.0 * (noise_pred_text_no_mask - noise_pred_uncond_no_mask) + guidance_scale_con * (noise_pred_both_no_mask - noise_pred_text_no_mask) \
                #     + 1.5 * (noise_pred_both_audio_no_mask - noise_pred_both_no_mask)
                #     noise_pred_uncond_mask, noise_pred_text_mask, noise_pred_both_mask, noise_pred_both_audio_mask = noise_pred_uncond[:,:,323:], noise_pred_text[:,:,323:], noise_pred_both[:,:,323:], noise_pred_both_audio[:,:,323:]
                #     noise_pred_mask = noise_pred_uncond_mask + 7.0 * (noise_pred_text_mask - noise_pred_uncond_mask) + guidance_scale_con * (noise_pred_both_mask - noise_pred_text_mask) \
                #     + 4.5 * (noise_pred_both_audio_mask - noise_pred_both_mask)
                #     noise_pred = torch.concat((noise_pred_no_mask, noise_pred_mask), dim=2)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 9. Post-processing
        if not output_type == "latent":
            with autocast():
                audio = self.vae.decode(latents).sample
        else:
            return AudioPipelineOutput(audios=latents)

        audio = audio[:, :, waveform_start:waveform_end]

        if output_type == "np":
            audio = audio.cpu().float().numpy()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)


def load_audio_file(filename, target_sr=44100, target_samples=2097152):
    try:
        audio, in_sr = torchaudio.load(filename)    
        # Resample if necessary
        if in_sr != target_sr:
            resampler = T.Resample(in_sr, target_sr)
            audio = resampler(audio)
        augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        audio = augs(audio)
        audio = audio.clamp(-1, 1)
        encoding = torch.nn.Sequential(
            Stereo(),
        )
        audio = encoding(audio)
        # audio.shape is [channels, samples]
        num_samples = audio.shape[-1]

        # if num_samples < target_samples:
        #     # Pad if it's too short
        #     pad_amount = target_samples - num_samples
        #     # Zero-pad at the end (or randomly if you prefer)
        #     audio = F.pad(audio, (0, pad_amount)) 
        #     print(f"pad {pad_amount}")
        # else:
        audio = audio[:, :target_samples]
        return audio
    except RuntimeError:
        print(f"Failed to decode audio file: {filename}")
        return None


class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            offset,
            offset + self.n_samples,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

# For zero initialized 1D CNN in the attention processor
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# Original attention processor for 
class StableAudioAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed 
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

            query = torch.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

                key = torch.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # print("hidden_states", hidden_states.shape)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
# The attention processor used in MuseControlLite, using 1 decoupled cross-attention layer
class StableAudioAttnProcessor2_0_rotary(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        ip_hidden_states = encoder_hidden_states_con
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions
        ###############################################################
        ip_hidden_states_output = None
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_key_length = ip_key.shape[2]
            ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            if kv_heads != attn.heads:
                # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
                heads_per_kv_head = attn.heads // kv_heads
                ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
                ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
            ip_value_length = ip_value.shape[2]
            seq_len_query = query.shape[2]

            # Generate position_ids for query, keys, values
            position_ids_query = torch.arange(
                seq_len_query, dtype=torch.long, device=query.device
            ) * (ip_key_length / seq_len_query)
            position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)
            position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
            position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)
            position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
            position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)
            
            # Rotate query, keys, values 
            cos, sin = self.rotary_emb(query, position_ids_query)
            query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
            cos, sin = self.rotary_emb(ip_key, position_ids_key)
            ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
            cos, sin = self.rotary_emb(ip_value, position_ids_value)
            ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
            
            ip_hidden_states = F.scaled_dot_product_attention(
                    query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
            ip_hidden_states = ip_hidden_states.transpose(1, 2)
            ip_hidden_states = self.conv_out(ip_hidden_states)
            ip_hidden_states_output = ip_hidden_states.transpose(1, 2)
        ###############################################################

        # Combine the output of the two cross-attention layers
        if ip_hidden_states_output is not None:
            hidden_states = hidden_states + self.scale * ip_hidden_states_output
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
# The attention processor used in MuseControlLite, using 2 decoupled cross-attention layer. It needs further examination, don't use it now.
class StableAudioAttnProcessor2_0_rotary_double(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.layer_id = layer_id
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.conv_out_audio = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.rotary_emb = LlamaRotaryEmbedding(64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
        # Below is for copying the weight of the original weight to the decoupled cross-attention
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # if self.layer_id == "0":
        #     hidden_states_sliced = hidden_states[:,1:,:]
        #     # Create a tensor of zeros with shape (bs, 1, 768)
        #     bs, _, dim2 = hidden_states_sliced.shape
        #     zeros = torch.zeros(bs, 1, dim2).cuda()
        #     # Concatenate the zero tensor along the second dimension (dim=1)
        #     hidden_states_sliced = torch.cat((hidden_states_sliced, zeros), dim=1)
        #     query_sliced = attn.to_q(hidden_states_sliced)
        #     query_sliced = query_sliced.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     query = query_sliced
        ip_hidden_states = encoder_hidden_states_con
        ip_hidden_states_audio = encoder_hidden_states_audio

        ip_hidden_states_out = None
        ip_hidden_states_audio_out = None

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_key_length = ip_key.shape[2]
            ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

            if kv_heads != attn.heads:
                heads_per_kv_head = attn.heads // kv_heads
                ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
                ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)

            ip_value_length = ip_value.shape[2]
            seq_len_query = query.shape[2]

            position_ids_query = torch.arange(
                seq_len_query, dtype=torch.long, device=query.device
            ) * (ip_key_length / seq_len_query)
            position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)
            position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
            position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)
            position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
            position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)

            cos, sin = self.rotary_emb(query, position_ids_query)
            query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))

            cos, sin = self.rotary_emb(ip_key, position_ids_key)
            ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))

            cos, sin = self.rotary_emb(ip_value, position_ids_value)
            ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))

            with torch.amp.autocast(device_type="cuda"):
                ip_hidden_states_out = F.scaled_dot_product_attention(
                    query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

            ip_hidden_states_out = ip_hidden_states_out.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states_out = ip_hidden_states_out.to(query.dtype)
            ip_hidden_states_out = ip_hidden_states_out.transpose(1, 2)

            with torch.amp.autocast(device_type="cuda"):
                ip_hidden_states_out = self.conv_out(ip_hidden_states_out)
            ip_hidden_states_out = ip_hidden_states_out.transpose(1, 2)

        if ip_hidden_states_audio is not None:
            ip_key_audio = self.to_k_ip_audio(ip_hidden_states_audio)
            ip_value_audio = self.to_v_ip_audio(ip_hidden_states_audio)
            ip_key_audio = ip_key_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            ip_key_audio_length = ip_key_audio.shape[2]
            ip_value_audio = ip_value_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

            if kv_heads != attn.heads:
                heads_per_kv_head = attn.heads // kv_heads
                ip_key_audio = torch.repeat_interleave(ip_key_audio, heads_per_kv_head, dim=1)
                ip_value_audio = torch.repeat_interleave(ip_value_audio, heads_per_kv_head, dim=1)

            seq_len_query = query.shape[2]
            ip_value_audio_length = ip_value_audio.shape[2]

            position_ids_query_audio = torch.arange(
                seq_len_query, dtype=torch.long, device=query.device
            ) * (ip_key_audio_length / seq_len_query)
            position_ids_query_audio = position_ids_query_audio.unsqueeze(0).expand(batch_size, -1)
            position_ids_key_audio = torch.arange(ip_key_audio_length, dtype=torch.long, device=key.device)
            position_ids_key_audio = position_ids_key_audio.unsqueeze(0).expand(batch_size, -1)
            position_ids_value_audio = torch.arange(
                ip_value_audio_length, dtype=torch.long, device=value.device
            )
            position_ids_value_audio = position_ids_value_audio.unsqueeze(0).expand(batch_size, -1)

            cos_audio, sin_audio = self.rotary_emb(query, position_ids_query_audio)
            query_pos_audio = (query * cos_audio.unsqueeze(1)) + (self.rotate_half(query) * sin_audio.unsqueeze(1))

            cos_audio, sin_audio = self.rotary_emb(ip_key_audio, position_ids_key_audio)
            ip_key_audio = (ip_key_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_key_audio) * sin_audio.unsqueeze(1))

            cos_audio, sin_audio = self.rotary_emb(ip_value_audio, position_ids_value_audio)
            ip_value_audio = (ip_value_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_value_audio) * sin_audio.unsqueeze(1))

            with torch.amp.autocast(device_type="cuda"):
                ip_hidden_states_audio_out = F.scaled_dot_product_attention(
                    query_pos_audio, ip_key_audio, ip_value_audio, attn_mask=None, dropout_p=0.0, is_causal=False
                )

            ip_hidden_states_audio_out = ip_hidden_states_audio_out.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states_audio_out = ip_hidden_states_audio_out.to(query.dtype)
            ip_hidden_states_audio_out = ip_hidden_states_audio_out.transpose(1, 2)

            with torch.amp.autocast(device_type="cuda"):
                ip_hidden_states_audio_out = self.conv_out_audio(ip_hidden_states_audio_out)
            ip_hidden_states_audio_out = ip_hidden_states_audio_out.transpose(1, 2)

        if ip_hidden_states_out is not None:
            hidden_states = hidden_states + self.scale * ip_hidden_states_out
        if ip_hidden_states_audio_out is not None:
            hidden_states = hidden_states + ip_hidden_states_audio_out
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
def setup_MuseControlLite(config, weight_dtype, transformer_ckpt):
    """
    Setup AP-adapter pipeline with attention processors and load checkpoints.
    
    Args:
        config: Configuration dictionary
        weight_dtype: Weight data type for the pipeline
        transformer_ckpt: Path to transformer checkpoint    
    Returns:
        tuple: (pipe, transformer) - Configured pipeline and transformer
    """
    # if 'audio' in config['condition_type'] and len(config['condition_type'])!=1:
        # from .lite_util_audio import StableAudioPipeline
    attn_processor = StableAudioAttnProcessor2_0_rotary_double
    audio_state_dict = load_file(config["audio_transformer_ckpt"], device="cpu")

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
    pipe.scheduler.config.sigma_max = config["sigma_max"]
    pipe.scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipe.transformer
    attn_procs = {}
    for name in transformer.attn_processors.keys():
        if name.endswith("attn1.processor"):
            attn_procs[name] = StableAudioAttnProcessor2_0()
        else:
            attn_procs[name] = StableAudioAttnProcessor2_0_rotary_double(
                layer_id = name.split(".")[1],
                hidden_size=768,
                name=name,
                cross_attention_dim=768,
                scale=config['ap_scale'],
            ).to("cuda", dtype=torch.float)            
    if transformer_ckpt is not None:
        state_dict = load_file(transformer_ckpt, device="cuda")
        for name, processor in attn_procs.items():
            if isinstance(processor, StableAudioAttnProcessor2_0_rotary_double):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                conv_out_weight = name + ".conv_out.weight"
                processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
                # if attn_processor == StableAudioAttnProcessor2_0_rotary_double:
                audio_weight_name_v = name + ".to_v_ip.weight"
                audio_weight_name_k = name + ".to_k_ip.weight"
                audio_conv_out_weight = name + ".conv_out.weight"
                processor.to_v_ip_audio.weight = torch.nn.Parameter(audio_state_dict[audio_weight_name_v].to(torch.float32))
                processor.to_k_ip_audio.weight = torch.nn.Parameter(audio_state_dict[audio_weight_name_k].to(torch.float32))
                processor.conv_out_audio.weight = torch.nn.Parameter(audio_state_dict[audio_conv_out_weight].to(torch.float32))
    transformer.set_attn_processor(attn_procs)
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipe.transformer(*args, **kwargs)
    transformer = _Wrapper(pipe.transformer.attn_processors)
    
    return pipe
def initialize_condition_extractors(config):
    """
    Initialize condition extractors based on configuration.
    
    Args:
        config: Configuration dictionary containing condition types and checkpoint paths
        
    Returns:
        tuple: (condition_extractors, transformer_ckpt, extractor_ckpt)
    """
    condition_extractors = {}
    extractor_ckpt = {}
    # from utils.feature_extractor import dynamics_extractor, rhythm_extractor, melody_extractor_mono, melody_extractor_stereo, melody_extractor_full_mono, melody_extractor_full_stereo, dynamics_extractor_full_stereo
    if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
        if "melody_stereo" in config['condition_type']:
            transformer_ckpt = config['transformer_ckpt_melody_stero']
            extractor_ckpt = config['extractor_ckpt_melody_stero']
            print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
            melody_conditoner = melody_extractor_full_stereo().cuda().float()
            condition_extractors["melody"] = melody_conditoner
        elif "melody_mono" in config['condition_type']:
            transformer_ckpt = config['transformer_ckpt_melody_mono']
            extractor_ckpt = config['extractor_ckpt_melody_mono']
            print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
            melody_conditoner = melody_extractor_full_mono().cuda().float()
            condition_extractors["melody"] = melody_conditoner
        elif "audio" in config['condition_type']:
            transformer_ckpt = config['audio_transformer_ckpt']
            print(f"using model: {transformer_ckpt}")
    else:
        dynamics_conditoner = dynamics_extractor().cuda().float()
        condition_extractors["dynamics"] = dynamics_conditoner
        rhythm_conditoner = rhythm_extractor().cuda().float()
        condition_extractors["rhythm"] = rhythm_conditoner
        melody_conditoner = melody_extractor_mono().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        transformer_ckpt = config['transformer_ckpt_musical']
        extractor_ckpt = config['extractor_ckpt_musical']
        print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
    
    for conditioner_type, ckpt_path in extractor_ckpt.items(): 
        state_dict = load_file(ckpt_path, device="cpu")
        condition_extractors[conditioner_type].load_state_dict(state_dict)
        condition_extractors[conditioner_type].eval()
    
    return condition_extractors, transformer_ckpt
def evaluate_and_plot_results(audio_file, gen_file_path, output_dir, i):
    """
    Evaluate and plot results comparing original and generated audio.
    
    Args:
        audio_file (str): Path to the original audio file
        gen_file_path (str): Path to the generated audio file
        output_dir (str): Directory to save the plot
        i (int): Index for naming the output file
    
    Returns:
        tuple: (dynamics_score, rhythm_score, melody_score)
    """

    dynamics_condition = compute_dynamics(audio_file)
    gen_dynamics = compute_dynamics(gen_file_path)
    min_len_dynamics = min(gen_dynamics.shape[0], dynamics_condition.shape[0])
    pearson_corr = np.corrcoef(gen_dynamics[:min_len_dynamics], dynamics_condition[:min_len_dynamics])[0, 1]
    print("pearson_corr", pearson_corr)
    
    melody_condition = extract_melody_one_hot(audio_file)      
    gen_melody = extract_melody_one_hot(gen_file_path)
    min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
    matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    print("melody accuracy", accuracy)
    
    # Adjust layout to avoid overlap
    processor = RNNDownBeatProcessor()
    original_path = os.path.join(output_dir, f"original_{i}.wav")
    input_probabilities = processor(original_path)
    generated_probabilities = processor(gen_file_path)
    hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
    input_timestamps = hmm_processor(input_probabilities)
    generated_timestamps = hmm_processor(generated_probabilities)
    precision, recall, f1 = evaluate_f1_rhythm(input_timestamps, generated_timestamps)
    # Output results
    print(f"F1 Score: {f1:.2f}")
    
    # Plotting
    frame_rate = 100  # Frames per second
    input_time_axis = np.linspace(0, len(input_probabilities) / frame_rate, len(input_probabilities))
    generate_time_axis = np.linspace(0, len(generated_probabilities) / frame_rate, len(generated_probabilities))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed

    # ----------------------------
    # Subplot (0,0): Dynamics Plot
    ax = axes[0, 0]
    ax.plot(dynamics_condition[:min_len_dynamics].squeeze(), linewidth=1, label='Dynamics condition')
    ax.set_title('Dynamics')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Dynamics (dB)')
    ax.legend(fontsize=8)
    ax.grid(True)
    # ----------------------------
    # Subplot (0,0): Dynamics Plot
    ax = axes[1, 0]
    ax.plot(gen_dynamics[:min_len_dynamics].squeeze(), linewidth=1, label='Generated Dynamics')
    ax.set_title('Dynamics')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Dynamics (dB)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # ----------------------------
    # Subplot (0,2): Melody Condition (Chromagram)
    ax = axes[0, 1]
    im2 = ax.imshow(melody_condition[:, :min_len_melody], aspect='auto', origin='lower',
                    interpolation='nearest', cmap='plasma')
    ax.set_title('Melody Condition')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chroma Features')

    # ----------------------------
    # Subplot (0,1): Generated Melody (Chromagram)
    ax = axes[1, 1]
    im1 = ax.imshow(gen_melody[:, :min_len_melody], aspect='auto', origin='lower',
                    interpolation='nearest', cmap='viridis')
    ax.set_title('Generated Melody')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chroma Features')

    # ----------------------------
    # Subplot (1,0): Rhythm Input Probabilities
    ax = axes[0, 2]
    ax.plot(input_time_axis, input_probabilities,
            label="Input Beat Probability")
    ax.plot(input_time_axis, input_probabilities,
            label="Input Downbeat Probability", alpha=0.8)
    ax.set_title('Rhythm: Input')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)

    # ----------------------------
    # Subplot (1,1): Rhythm Generated Probabilities
    ax = axes[1, 2]
    ax.plot(generate_time_axis, generated_probabilities,
            color='orange', label="Generated Beat Probability")
    ax.plot(generate_time_axis, generated_probabilities,
            alpha=0.8, color='red', label="Generated Downbeat Probability")
    ax.set_title('Rhythm: Generated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)

    # Adjust layout and save the combined image
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"combined_{i}.png")
    plt.savefig(combined_path)
    plt.close()

    print(f"Combined plot saved to {combined_path}")
    
    return pearson_corr, f1, accuracy

def process_musical_conditions(config, audio_file, condition_extractors, weight_dtype, MuseControlLite):
    """
    Process and extract musical conditions (dynamics, rhythm, melody) from audio file.
    
    Args:
        config: Configuration dictionary
        audio_file: Path to the audio file
        condition_extractors: Dictionary of condition extractors
        output_dir: Output directory path
        i: Index for file naming
        weight_dtype: Weight data type for torch tensors
        MuseControlLite: The MuseControlLite model instance
        audio_mask_start: Start index for audio mask
        audio_mask_end: End index for audio mask
        musical_attribute_mask_start: Start index for musical attribute mask
        musical_attribute_mask_end: End index for musical attribute mask
    
    Returns:
        tuple: (final_condition, extracted_condition, final_condition_audio)
    """
    total_seconds = 2097152/44100
    use_audio_mask = False
    use_musical_attribute_mask = False
    if (config["audio_mask_start_seconds"] and config["audio_mask_end_seconds"]) != 0 and "audio" in config["condition_type"]:
        use_audio_mask = True
        audio_mask_start = int(config["audio_mask_start_seconds"] / total_seconds * 1024) # 1024 is the latent length for 2097152/44100 seconds
        audio_mask_end = int(config["audio_mask_end_seconds"] / total_seconds * 1024)
        print(
            f"using mask for 'audio' from "
            f"{config['audio_mask_start_seconds']}~{config['audio_mask_end_seconds']}"
        )
    if (config["musical_attribute_mask_start_seconds"] and config["musical_attribute_mask_end_seconds"]) != 0:
        use_musical_attribute_mask = True
        musical_attribute_mask_start = int(config["musical_attribute_mask_start_seconds"] / total_seconds * 1024)
        musical_attribute_mask_end = int(config["musical_attribute_mask_end_seconds"] / total_seconds * 1024)
        masked_types = [t for t in config['condition_type'] if t != 'audio']
        print(
            f"using mask for {', '.join(masked_types)} "
            f"from {config['musical_attribute_mask_start_seconds']}~"
            f"{config['musical_attribute_mask_end_seconds']}"
        )
    if "dynamics" in config["condition_type"]:
        dynamics_condition = compute_dynamics(audio_file)
        dynamics_condition = torch.from_numpy(dynamics_condition).cuda()
        dynamics_condition = dynamics_condition.unsqueeze(0).unsqueeze(0)
        print("dynamics_condition", dynamics_condition.shape)
        extracted_dynamics_condition = condition_extractors["dynamics"](dynamics_condition.to(torch.float32))
        masked_extracted_dynamics_condition =  torch.zeros_like(extracted_dynamics_condition)
        extracted_dynamics_condition = F.interpolate(extracted_dynamics_condition, size=1024, mode='linear', align_corners=False) 
        masked_extracted_dynamics_condition = F.interpolate(masked_extracted_dynamics_condition, size=1024, mode='linear', align_corners=False)
    else: 
        extracted_dynamics_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_dynamics_condition = extracted_dynamics_condition

    if "rhythm" in config["condition_type"]:
        rnn_processor = RNNDownBeatProcessor()
        wave = load_audio_file(audio_file)
        if wave is not None:
            # wave: torch.Tensor, shape [2, samples]，44100Hz，已經裁到 2097152
            # 直接建立 madmom 的 Signal（不落地存檔）
            # 注意：madmom 預設以單聲道處理；若想保留雙聲道可自行平均或取一邊
            # 這裡維持與原本處理邏輯一致（原本寫檔後由 madmom 自行讀入，實務上也會混成單聲道）
            wave_np = wave.mean(dim=0).float().cpu().numpy()  # (samples,) → 單聲道
            sig = Signal(wave_np, sample_rate=44100)

            # 直接把 Signal 餵給 RNNDownBeatProcessor（不需要檔案路徑）
            rhythm_curve = rnn_processor(sig)     # shape: (T, 2)，[beat_prob, downbeat_prob]
            rhythm_condition = torch.from_numpy(rhythm_curve).cuda().transpose(0, 1).unsqueeze(0)
            #                           ^ numpy -> torch            (2, T) -> (1, 2, T)

            print("rhythm_condition", rhythm_condition.shape)

            extracted_rhythm_condition = condition_extractors["rhythm"](rhythm_condition.to(torch.float32))
            masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
            extracted_extracted_size = 1024
            extracted_rhythm_condition = F.interpolate(
                extracted_rhythm_condition, size=extracted_extracted_size, mode='linear', align_corners=False
            )
            masked_extracted_rhythm_condition = F.interpolate(
                masked_extracted_rhythm_condition, size=extracted_extracted_size, mode='linear', align_corners=False
            )
        else:
            extracted_rhythm_condition = torch.zeros((1, 192, 1024), device="cuda")
            masked_extracted_rhythm_condition = extracted_rhythm_condition
    else: 
        extracted_rhythm_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_rhythm_condition = extracted_rhythm_condition
    
    if "melody_mono" in config["condition_type"]:
        melody_condition = compute_melody(audio_file)
        melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
        print("melody_condition", melody_condition.shape)
        extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
    elif "melody_stereo" in config["condition_type"]:
        melody_condition = compute_melody_v2(audio_file)
        melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
        print("melody_condition", melody_condition.shape)
        extracted_melody_condition = condition_extractors["melody"](melody_condition)
        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
    else: 
        if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
            extracted_melody_condition = torch.zeros((1, 768, 1024), device="cuda")
        else:
            extracted_melody_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_melody_condition = extracted_melody_condition

    # Use multiple cfg
    if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
        extracted_condition = extracted_melody_condition
        final_condition = torch.concat((masked_extracted_melody_condition, masked_extracted_melody_condition, extracted_melody_condition), dim=0)
    else:
        extracted_blank_condition = torch.zeros((1, 192, 1024), device="cuda")
        extracted_condition = torch.concat((extracted_rhythm_condition, extracted_dynamics_condition, extracted_melody_condition, extracted_blank_condition), dim=1)
        masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_dynamics_condition, masked_extracted_melody_condition, extracted_blank_condition), dim=1)
        final_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
    if "audio" in config["condition_type"]:
        desired_repeats = 768 // 64  # Number of repeats needed
        audio = load_audio_file(audio_file)
        if audio is not None:
            audio_condition = MuseControlLite.vae.encode(audio.unsqueeze(0).to(weight_dtype).cuda()).latent_dist.sample()
            extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1).float()
            pad_len = 1024 - extracted_audio_condition.shape[-1]
            if pad_len > 0:
                # Pad on the right side (last dimension)
                extracted_audio_condition = F.pad(extracted_audio_condition, (0, pad_len)) 
            masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
            if len(config["condition_type"]) == 1:
                final_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
            else:
                final_condition_audio = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
                final_condition = torch.concat((final_condition, extracted_condition), dim=0)
                final_condition_audio = final_condition_audio.transpose(1, 2)
        else:
            final_condition_audio = None
    final_condition = final_condition.transpose(1, 2)
    if "audio" in config["condition_type"] and len(config["condition_type"])==1:
        final_condition[:,audio_mask_start:audio_mask_end,:] = 0
        if use_audio_mask:
            config["guidance_scale_con"] = config["guidance_scale_audio"]
    elif "audio" in config["condition_type"] and len(config["condition_type"])!=1 and use_audio_mask:
        final_condition[:,:audio_mask_start,:] = 0
        final_condition[:,audio_mask_end:,:] = 0
        if 'final_condition_audio' in locals() and final_condition_audio is not None:
            final_condition_audio[:,audio_mask_start:audio_mask_end,:] = 0
    elif use_musical_attribute_mask:
        final_condition[:,musical_attribute_mask_start:musical_attribute_mask_end,:] = 0
        if 'final_condition_audio' in locals() and final_condition_audio is not None:
            final_condition_audio[:,:musical_attribute_mask_start,:] = 0
            final_condition_audio[:,musical_attribute_mask_end:,:] = 0
    
    return final_condition, final_condition_audio if 'final_condition_audio' in locals() else None

import torchaudio
import numpy as np
from scipy.signal import savgol_filter
import librosa
import torch
import torchaudio
import scipy.signal as signal
from torchaudio import transforms as T
import torch
import torchaudio
import librosa
import numpy as np


def compute_melody_v2(stereo_audio: torch.Tensor) -> np.ndarray:
    """
    Args:
        stereo_audio: torch.Tensor of shape (2, N), 其中 stereo_audio[0] 是左聲道,
                      stereo_audio[1] 是右聲道。
        sr:           取樣率 (sampling rate)。
    Returns:
        c: np.ndarray of shape (8, T_frames)，
           每一列代表： [L1, R1, L2, R2, L3, R3, L4, R4]（按 frame 交錯），
           且每個值都 ∈ {1, 2, …, 128}，對應 CQT 的頻率 bin。
    """
    audio, sr = torchaudio.load(stereo_audio)
    # 1. 先針對左、右聲道分別計算 CQT (128 bins)，回傳 cqt_db 形狀都是 (128, T_frames)
    cqt_left  = compute_music_represent(audio[0], sr)  # shape: (128, T_frames)
    cqt_right = compute_music_represent(audio[1], sr)  # shape: (128, T_frames)

    # 2. 取得時框 (frame) 數量
    #    注意：librosa.cqt 的輸出 cqt_db 對應的「時框數」就是第二維度
    T_frames = cqt_left.shape[1]

    # 3. 預先配置輸出矩陣 c，dtype 用 int，shape = (8, T_frames)
    c = np.zeros((8, T_frames), dtype=np.int32)

    # 4. 逐一 frame 處理：對每個 frame 的 128 維度做 top-4
    for j in range(T_frames):
        # 4.1 取出當前時框的左、右聲道 CQT 能量（分貝值）
        col_L = cqt_left[:, j]   # shape: (128,)
        col_R = cqt_right[:, j]  # shape: (128,)

        # 4.2 用 numpy.argsort 找到「前 4 大」的索引
        #     np.argsort 預設是從小到大排序，所以取最後 4 個，再反轉取大到小
        idx4_L = np.argsort(col_L)[-4:][::-1]  # 0-based, 長度=4
        idx4_R = np.argsort(col_R)[-4:][::-1]  # 0-based, 長度=4

        # 4.3 轉成 1-based（因為題意寫 pixel ∈ {1,2,…,128}）
        idx4_L = idx4_L + 1  # 現在範圍是 1..128
        idx4_R = idx4_R + 1

        # 4.4 交錯填入 c 的第 j 欄
        #     我們希望 c[:, j] = [L1, R1, L2, R2, L3, R3, L4, R4]
        for k in range(4):
            c[2 * k    , j] = idx4_L[k]
            c[2 * k + 1, j] = idx4_R[k]

    return c[:,:4097]


def compute_music_represent(audio, sr):
    filter_y = torchaudio.functional.highpass_biquad(audio, sr, 261.6)
    fmin = librosa.midi_to_hz(0)
    cqt_spec = librosa.cqt(y=filter_y.numpy(), fmin=fmin, sr=sr, n_bins=128, bins_per_octave=12, hop_length=512)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    return cqt_db

def keep_top4_pitches_per_channel(cqt_db):
    """
    cqt_db is assumed to have shape: (2, 128, time_frames).
    We return a combined 2D array of shape (128, time_frames)
    where only the top 4 pitch bins in each channel are kept
    (for a total of up to 8 bins per time frame).
    """
    # Parse shapes
    num_channels, num_bins, num_frames = cqt_db.shape
    
    # Initialize an output array that combines both channels
    # and has zeros everywhere initially
    combined = np.zeros((num_bins, num_frames), dtype=cqt_db.dtype)
    
    for ch in range(num_channels):
        for t in range(num_frames):
            # Find the top 4 pitch bins for this channel at frame t
            # argsort sorts ascending; we take the last 4 indices for top 4
            top4_indices = np.argsort(cqt_db[ch, :, t])[-4:]
            
            # Copy their values into the combined array
            # We add to it in case there's overlap between channels
            combined[top4_indices, t] = 1
    return combined
def compute_melody(input_audio):
    # Initialize parameters
    sample_rate = 44100

    # Load audio file
    wav, sr = torchaudio.load(input_audio)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = resample(wav)
    # Truncate or pad the audio to 2097152 samples
    target_length = 2097152
    if wav.size(1) > target_length:
        # Truncate the audio if it is longer than the target length
        wav = wav[:, :target_length]
    elif wav.size(1) < target_length:
        # Pad the audio with zeros if it is shorter than the target length
        padding = target_length - wav.size(1)
        wav = torch.cat([wav, torch.zeros(wav.size(0), padding)], dim=1)
    melody = compute_music_represent(wav, 44100)
    melody = keep_top4_pitches_per_channel(melody)    
    return melody

def compute_dynamics(audio_file, hop_length=160, target_sample_rate=44100, cut=True):
    """
    Compute the dynamics curve for a given audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        window_length (int): Length of FFT window for computing the spectrogram.
        hop_length (int): Number of samples between successive frames.
        smoothing_window (int): Length of the Savitzky-Golay filter window.
        polyorder (int): Polynomial order of the Savitzky-Golay filter.

    Returns:
        dynamics_curve (numpy.ndarray): The computed dynamic values in dB.
    """
    # Load audio file
    waveform, original_sample_rate = torchaudio.load(audio_file)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if cut:
        waveform = waveform[:, :2097152]
    # Ensure waveform has a single channel (e.g., select the first channel if multi-channel)
    waveform = waveform.mean(dim=0, keepdim=True)  # Mix all channels into one
    waveform = waveform.clamp(-1, 1).numpy()
    
    S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=hop_length))
    mel_filter_bank = librosa.filters.mel(sr=target_sample_rate, n_fft=1024, n_mels=64, fmin=0, fmax=8000)
    S = np.dot(mel_filter_bank, S)
    energy = np.sum(S**2, axis=0)
    dynamics_db = np.clip(energy, 1e-6, None)
    dynamics_db = librosa.amplitude_to_db(energy, ref=np.max).squeeze(0)
    smoothed_dynamics = savgol_filter(dynamics_db, window_length=279, polyorder=1)
    # print(smoothed_dynamics.shape)
    return smoothed_dynamics
def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256):
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma
def evaluate_f1_rhythm(input_timestamps, generated_timestamps, tolerance=0.07):
    """
    Evaluates precision, recall, and F1-score for beat/downbeat timestamp alignment.
    
    Args:
        input_timestamps (ndarray): 2D array of shape [n, 2], where column 0 contains timestamps.
        generated_timestamps (ndarray): 2D array of shape [m, 2], where column 0 contains timestamps.
        tolerance (float): Alignment tolerance in seconds (default: 70ms).
    
    Returns:
        tuple: (precision, recall, f1)
    """
    # Extract and sort timestamps
    input_timestamps = np.asarray(input_timestamps)
    generated_timestamps = np.asarray(generated_timestamps)
    
    # If you only need the first column
    if input_timestamps.size > 0:  
        input_timestamps = input_timestamps[:, 0]
        input_timestamps.sort()
    else:
        input_timestamps = np.array([])
        
    if generated_timestamps.size > 0:
        generated_timestamps = generated_timestamps[:, 0]
        generated_timestamps.sort()
    else:
        generated_timestamps = np.array([])

    # Handle empty cases
    # Case 1: Both are empty
    if len(input_timestamps) == 0 and len(generated_timestamps) == 0:
        # You could argue everything is correct since there's nothing to detect,
        # but returning all zeros is a common convention.
        return 0.0, 0.0, 0.0

    # Case 2: No ground-truth timestamps, but predictions exist
    if len(input_timestamps) == 0 and len(generated_timestamps) > 0:
        # All predictions are false positives => tp=0, fp = len(generated_timestamps)
        # => precision=0, recall is undefined (tp+fn=0), typically we treat recall=0
        return 0.0, 0.0, 0.0

    # Case 3: Ground-truth timestamps exist, but no predictions
    if len(input_timestamps) > 0 and len(generated_timestamps) == 0:
        # Everything in input_timestamps is a false negative => tp=0, fn = len(input_timestamps)
        # => recall=0, precision is undefined (tp+fp=0), typically we treat precision=0
        return 0.0, 0.0, 0.0

    # If we get here, both arrays are non-empty
    tp = 0
    fp = 0
    
    # Track matched ground-truth timestamps
    matched_inputs = np.zeros(len(input_timestamps), dtype=bool)
    
    for gen_ts in generated_timestamps:
        # Calculate absolute differences to each reference timestamp
        diffs = np.abs(input_timestamps - gen_ts)
        # Find index of the closest input timestamp
        min_diff_idx = np.argmin(diffs)
        
        # Check if that difference is within tolerance and unmatched
        if diffs[min_diff_idx] < tolerance and not matched_inputs[min_diff_idx]:
            tp += 1
            matched_inputs[min_diff_idx] = True
        else:
            fp += 1  # no suitable match found or closest was already matched
    
    # Remaining unmatched input timestamps are false negatives
    fn = np.sum(~matched_inputs)
    
    # Compute precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


import torch.nn as nn
import torch.nn.functional as F

class dynamics_extractor_full_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(2, 16, kernel_size=3, padding=1, stride=2)  
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2)
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 1, 8280)
        # x = x.unsqueeze(1) # shape: (batchsize, 1, 8280)
        x = self.conv1d_1(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_2(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_3(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_4(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_5(x)  # shape: (batchsize, 192, 1035)
        return x
class melody_extractor_full_mono(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(128, 256, kernel_size=3, padding=0, stride=2)  
        self.conv1d_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2)  
        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(512, 768, kernel_size=3, padding=1)
    def forward(self, x):
        # original shape: (batchsize, 12, 1296)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1024)
        x = F.silu(x)
        x = self.conv1d_4(x) # shape: (batchsize, 128, 1024)
        x = F.silu(x)
        x = self.conv1d_5(x) # shape: (batchsize, 768, 1024)
        return x
class melody_extractor_mono(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(128, 128, kernel_size=3, padding=0, stride=2)  
        self.conv1d_2 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)  
        self.conv1d_3 = nn.Conv1d(192, 192, kernel_size=3, padding=1)
    def forward(self, x):
        # original shape: (batchsize, 12, 1296)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1024)
        return x
    
class melody_extractor_full_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=129, embedding_dim=48)

        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(384, 384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(384, 768, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(768, 768, kernel_size=3, padding=1)

    def forward(self, melody_idxs):
        # melody_idxs: LongTensor of shape (B, 8, 4096)
        B, eight, L = melody_idxs.shape  # L == 4096

        # 1) Embed:
        #    (B, 8, 4096) → (B, 8, 4096, 48)
        embedded = self.embed(melody_idxs)

        # 2) Permute & reshape → (B, 8*48, 4096) = (B, 384, 4096)
        x = embedded.permute(0, 1, 3, 2)      # (B, 8, 48, 4096)
        x = x.reshape(B, eight * 48, L)       # (B, 384, 4096)

        # 3) Conv1 → (B, 384, 4096)
        x = F.silu(self.conv1(x))

        # 4) Conv2 → (B, 768, 4096)
        x = F.silu(self.conv2(x))

        # 5) Conv3 → (B, 768, 4096)
        x = F.silu(self.conv3(x))

        # Now x is (B, 1536, 4096) and can be sent on to whatever comes next
        return x
class melody_extractor_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=129, embedding_dim=4)

        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

    def forward(self, melody_idxs):
        # melody_idxs: LongTensor of shape (B, 8, 4096)
        B, eight, L = melody_idxs.shape  # L == 4096

        # 1) Embed:
        #    (B, 8, 4096) → (B, 8, 4096, 4)
        embedded = self.embed(melody_idxs)

        # 2) Permute & reshape → (B, 8*4, 4096) = (B, 32, 4096)
        x = embedded.permute(0, 1, 3, 2)      # (B, 8, 4, 4096)
        x = x.reshape(B, eight * 4, L)       # (B, 32, 4096)

        # 3) Conv1 → (B, 384, 4096)
        x = F.silu(self.conv1(x))

        # 4) Conv2 → (B, 768, 4096)
        x = F.silu(self.conv2(x))

        # 5) Conv3 → (B, 768, 4096)
        x = F.silu(self.conv3(x))

        x = F.silu(self.conv4(x))

        x = F.silu(self.conv5(x))

        # Now x is (B, 1536, 4096) and can be sent on to whatever comes next
        return x

class dynamics_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=2)  
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2)
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 1, 8280)
        # x = x.unsqueeze(1) # shape: (batchsize, 1, 8280)
        x = self.conv1d_1(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_2(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_3(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_4(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_5(x)  # shape: (batchsize, 192, 1035)
        return x
class rhythm_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)  
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 2, 3000)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_4(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_5(x) # shape:
        return x
    
    
import torch
import soundfile as sf
# from MuseControlLite_setup import (
#     setup_MuseControlLite,
#     initialize_condition_extractors,
#     evaluate_and_plot_results,
#     load_audio_file,
#     process_musical_conditions
# )
import os
import numpy as np
# from config_inference import get_config
import argparse
import json

def get_config():
    return {
        "condition_type": ["melody_stereo"], #  you can choose any combinations in the two sets: ["dynamics", "rhythm", "melody_mono", "audio"],  ["melody_stereo", "audio"]
                                    # When using audio, is recommend to use empty string "" as prompt
        "output_dir": "./generated_audio/output",

        "GPU_id": "0",

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for Musical attribute condition
        
        "guidance_scale_audio": 1.0,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp16", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "",

        ###############

        "audio_mask_start_seconds": 14, # Apply mask to musical attributes choose only one mask to use, it automatically generates a complemetary mask to the other condition

        "audio_mask_end_seconds": 47, 

        "musical_attribute_mask_start_seconds": 0, # 'Apply mask to audio condition, choose only one mask to use, it automatically generates a complemetary mask to the other condition'

        "musical_attribute_mask_end_seconds": 0,

        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "show_result_and_plt": True,

        "audio_files": [
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/610_bass.mp3",
            "melody_condition_audio/610_bass.mp3",
            "melody_condition_audio/785_piano.mp3",
            "melody_condition_audio/785_piano.mp3",
            "melody_condition_audio/933_string.mp3",
            "melody_condition_audio/933_string.mp3",
            "melody_condition_audio/6_uke_12.wav",
            "melody_condition_audio/6_uke_12.wav",
            "melody_condition_audio/57_jazz.mp3",
            "melody_condition_audio/703_mideast.mp3",

        ],
        # "audio_files": [
        #     "SDD_nosinging/SDD_audio/34/1004034.mp3",
        #     "original_15s/original_9.wav",
        #     "original_15s/original_10.wav",
        #     "original_15s/original_11.wav",
        #     "original_15s/original_15.wav",
        #     "original_15s/original_16.wav",
        #     "original_15s/original_21.wav",
        #     "original_15s/original_25.wav",
        # ],

        "text": [
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "A heartfelt, warm acoustic guitar performance, evoking a sense of tenderness and deep emotion, with a melody that truly resonates and touches the heart.",     
                "A vibrant MIDI electronic composition with a hopeful and optimistic vibe.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "Hopeful instrumental with guitar being the lead and tabla used for percussion in the middle giving a feeling of going somewhere with positive outlook.",
                "A string ensemble opens the track with legato, melancholic melodies. The violins and violas play beautifully, while the cellos and bass provide harmonic support for the moving passages. The overall feel is deeply melancholic, with an emotionally stirring performance that remains harmonious and a sense of clearness.",
                "An exceptionally harmonious string performance with a lively tempo in the first half, transitioning to a gentle and beautiful melody in the second half. It creates a warm and comforting atmosphere, featuring cellos and bass providing a solid foundation, while violins and violas showcase the main theme, all without any noise, resulting in a cohesive and serene sound.",
                "Pop solo piano instrumental song. Simple harmony and emotional theme. Makes you feel nostalgic and wanting a cup of warm tea sitting on the couch while holding the person you love.",
                "A whimsical string arrangement with rich layers, featuring violins as the main melody, accompanied by violas and cellos. The light, playful melody blends harmoniously, creating a sense of clarity.",
                "An instrumental piece primarily featuring acoustic guitar, with a lively and nimble feel. The melody is bright, delivering an overall sense of joy.",
                "A joyful saxophone performance that is smooth and cohesive, accompanied by cello. The first half features a relaxed tempo, while the second half picks up with an upbeat rhythm, creating a lively and energetic atmosphere. The overall sound is harmonious and clear, evoking feelings of happiness and vitality.",
                "A cheerful piano performance with a smooth and flowing rhythm, evoking feelings of joy and vitality.",
                "An instrumental piece primarily featuring piano, with a lively rhythm and cheerful melodies that evoke a sense of joyful childhood playfulness. The melodies are clear and bright.",
                "fast and fun beat-based indie pop to set a protagonist-gets-good-at-x movie montage to.",
                "A lively 70s style British pop song featuring drums, electric guitars, and synth violin. The instruments blend harmoniously, creating a dynamic, clean sound without any noise or clutter.",
                "A soothing acoustic guitar song that evokes nostalgia, featuring intricate fingerpicking. The melody is both sacred and mysterious, with a rich texture."
                ],

        ########## adapters avilable ############
        # We trained 4 set of adapters:
        # 1. with conditions ["melody_mono", "dynamics", "rhythm"]
        # 2. with conditions ["melody_mono"]
        # 3. with conditions ["melody_stereo"]
        # 3. with conditions ["audio"]
        # MuseControlLite_inference_all.py will automaticaly choose the most suitable model according to the condition type:
        ###############
        # Works for condition ["dynamics", "rhythm", "melody_mono"]
        "transformer_ckpt_musical": "./checkpoints/woSDD-all/model_3.safetensors",
        
        "extractor_ckpt_musical": {
            "dynamics": "./checkpoints/woSDD-all/model_1.safetensors",
            "melody": "./checkpoints/woSDD-all/model.safetensors",
            "rhythm": "./checkpoints/woSDD-all/model_2.safetensors",
        },
        ###############

        # Works for ['audio], it works without a feature extractor, and could cooperate with other adapters
        #################
        "audio_transformer_ckpt": "./checkpoints/70000_Audio/model.safetensors",

        # Specialized for ['melody_stereo']
        ###############
        "transformer_ckpt_melody_stero": "./checkpoints/70000_Melody_stereo/model_1.safetensors",

        "extractor_ckpt_melody_stero": {
            "melody": "./checkpoints/70000_Melody_stereo/model.safetensors",
        },
        ###############

        # Specialized for ['melody_mono']
        ###############
        "transformer_ckpt_melody_mono": "./checkpoints/40000_Melody_mono/model_1.safetensors",

        "extractor_ckpt_melody_mono": {
            "melody": "./checkpoints/40000_Melody_mono/model.safetensors",
        },
        ###############
    }

class MusicControlLiteGenerator:
    
    def __init__(self):

        config = get_config()  # Pass the parsed arguments to get_config
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        self.weight_dtype = torch.float32
        if config["weight_dtype"] == "fp16":
            weight_dtype = torch.float16
        self.condition_extractors, transformer_ckpt = initialize_condition_extractors(config)
        MuseControlLite = setup_MuseControlLite(config, weight_dtype, transformer_ckpt)
        MuseControlLite = MuseControlLite.to("cuda")

        negative_text_prompt = config["negative_text_prompt"]
        # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
        score_dynamics = []
        score_rhythm = []
        score_melody = []
        
        self.pipe = MuseControlLite
        
    @torch.no_grad()
    def generate(self, prompt, audio_path):
        

        final_condition, final_condition_audio = process_musical_conditions(self.config, audio_path, self.condition_extractors, self.weight_dtype, self.pipe)
        # if config["no_text"] is True:
        #     prompt_texts = ""
        print("prompt_texts", prompt)
        
        negative_text_prompt = "Low quality."
        
        waveform = self.pipe(
            extracted_condition=final_condition, 
            extracted_condition_audio=final_condition_audio,
            prompt=prompt,
            negative_prompt=negative_text_prompt,
            num_inference_steps=self.config["denoise_step"],
            guidance_scale_text=self.config["guidance_scale_text"],
            guidance_scale_con=self.config["guidance_scale_con"],
            guidance_scale_audio=self.config["guidance_scale_audio"],
            num_waveforms_per_prompt=1,
            audio_end_in_s=2097152 / 44100,
            generator = torch.Generator().manual_seed(42)
        ).audios 
        # save audio
        print(waveform.shape)
        
        return waveform.cpu().squeeze().numpy().T.astype(np.float32), 44100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")

    generator = MusicControlLiteGenerator()
    music, sr = generator.generate(
        prompt="A happy melody",
        audio_path='Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav'
    )
    
    sf.write(os.path.join(f"generated_music.wav"), music, sr)
