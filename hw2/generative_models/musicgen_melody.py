from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, BitsAndBytesConfig
import torch, librosa

class MusicGenGenerator:
    def __init__(self,
                 model_name: str = "facebook/musicgen-melody",
                 quant: str = "4bit"):
        self.processor = AutoProcessor.from_pretrained(model_name)

        # 用 BF16（Ada/Lovelace 很友善）
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        bnb_config = None
        if quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quant == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda",                # ← 建議用 "auto"
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.sr = 32000

    @torch.no_grad()
    def generate(self, prompt, audio_path,
                 negative_prompt=None,
                 num_inference_steps=60,
                 audio_end_in_s=47.0,
                 guidance_scale=7.0):

        wav, _ = librosa.load(audio_path, sr=self.sr)

        inputs = self.processor(
            audio=wav,
            sampling_rate=self.sr,
            text=prompt,
            return_tensors="pt"
        )
        # → 統一 dtype：浮點數用 model.dtype，整數照搬
        casted = {}
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                casted[k] = v.to(self.model.device, dtype=self.model.dtype)
            else:
                casted[k] = v.to(self.model.device)

        gen = self.model.generate(
            **casted,
            do_sample=True,
            guidance_scale=guidance_scale,
            # num_inference_steps=num_inference_steps,
            # audio_end_in_s=audio_end_in_s,
        )

        # 可能是 bfloat16，存檔前轉成 float32 比較穩
        return gen.squeeze().to(torch.float32).cpu(), self.sr


if __name__ == "__main__":
    generator = MusicGenGenerator(quant="4bit")
    audio, sr = generator.generate(
        prompt="A happy melody",
        audio_path="Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav"
    )

    import torchaudio
    torchaudio.save("generated_music.wav", audio.cpu(), sr, encoding="PCM_S", bits_per_sample=16)