from transformers import pipeline
import torch, scipy

class MusicGenGenerator:
    def __init__(self, model_name="facebook/musicgen-medium"):
        self.pipe = pipeline(
            "text-to-audio",
            model=model_name,
            model_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": torch.float16,
            }
        )

    def generate(self, prompt, negative_prompt=None, num_inference_steps=60, audio_end_in_s=47.0, guidance_scale=7.0):
        music = self.pipe(
            prompt,
            forward_params={"do_sample": True}
        )
        music['audio'] = music['audio'].squeeze()
        return music["audio"], music["sampling_rate"]