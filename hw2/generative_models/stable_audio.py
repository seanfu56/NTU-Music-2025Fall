import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    StableAudioDiTModel, StableAudioPipeline
)
from transformers import BitsAndBytesConfig, T5EncoderModel
import soundfile as sf

class StableAudioGenerator:
    def __init__(self):
        # 1) 量化設定
        bnb_text = BitsAndBytesConfig(load_in_8bit=True)  # T5 用 8-bit
        bnb_dit = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,                # DiT 用 4-bit 更省顯存
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # 2) 逐一載入（不要給 device_map）
        text_encoder = T5EncoderModel.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            subfolder="text_encoder",
            quantization_config=bnb_text,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        transformer = StableAudioDiTModel.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            subfolder="transformer",
            quantization_config=bnb_dit,
            torch_dtype=torch.float16,
        )

        # 3) 組 pipeline（同樣不要給 device_map）
        self.pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.float16,
        )

        # 4) 一句話把整個 pipeline 搬到同一張 GPU（符合你的 CUDA_VISIBLE_DEVICES=1）
        self.pipe.to("cuda")

        # 可開的省顯存選項（不涉及 auto device_map）
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_slicing()
        # 若已安裝 xformers，可再省一截：
        # self.pipe.enable_xformers_memory_efficient_attention()

    def generate(self, prompt, negative_prompt="Low quality.", num_inference_steps=60, audio_end_in_s=47.0, guidance_scale=7.0, num_waveforms_per_prompt=1):
        # 5) 推理
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_end_in_s=audio_end_in_s,
            guidance_scale=guidance_scale,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
        )
        return out.audios[0].T.float().cpu().numpy(), 16000