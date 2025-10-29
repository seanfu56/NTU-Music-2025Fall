from transformers import AutoModel
import einops
import torch
from sklearn.metrics.pairwise import cosine_similarity


class AudioMAE:
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True).to(self.device)
        self.model.eval()
        
    def get_embedding(self, audio_path):
        with torch.no_grad():
            z = self.model(audio_path)  # (768, 8, 64) = (latent_dim_size, latent_freq_dim, latent_temporal_dim)
            z = einops.rearrange(z, 'd f t -> (f t) d')  # (8*64, 768)
            z = z.mean(dim=0)  # (768,)
            vector = z.cpu().numpy()
        return vector
    
if __name__ == "__main__":
    audio_mae = AudioMAE()
    embedding1 = audio_mae.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")
    # embedding2 = audio_mae.get_embedding("Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")
    
    embedding2 = audio_mae.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")
    

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)