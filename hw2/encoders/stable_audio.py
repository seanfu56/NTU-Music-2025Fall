# pip install -U diffusers torch soundfile librosa
import torch, librosa, numpy as np, soundfile as sf
from diffusers import AutoencoderOobleck


class StableAudio:
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        
        self.vae.eval()
        
    def get_embedding(self, audio_path):
        wav, sr = librosa.load(audio_path, sr=44_100, mono=False)
        if wav.ndim == 1:  # [T] -> [2, T]
            wav = np.stack([wav, wav], axis=0)
        wave = torch.from_numpy(wav).unsqueeze(0).to(self.device, dtype=torch.float16)  # [1,2,T], 值域約 [-1,1]
        
        with torch.no_grad():
            enc = self.vae.encode(wave)                   # -> 分布
            z = enc.latent_dist.mode()                     # 或 .sample()
            z = z.mean(dim=2)  # [1, C, T']
        return z.squeeze(0).cpu().numpy()  # [C, T']
    
if __name__ == "__main__":
    stable_audio = StableAudio()
    embedding1 = stable_audio.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")
    embedding2 = stable_audio.get_embedding("Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")
    
    # embedding2 = stable_audio.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")
    

    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)