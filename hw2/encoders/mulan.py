import torch
import librosa

from muq import MuQMuLan

class MuLan:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device)
        self.sample_rate = 32_000  # MuLan 使用 32kHz 音訊
        
        self.model.eval()

    def get_embedding(self, audio_path):
        audio = librosa.load(audio_path, sr=self.sample_rate)[0]
        inputs = torch.tensor(audio).unsqueeze(0).to(self.device)  # [1, T]
        with torch.no_grad():
            outputs = self.model(wavs=inputs)
        
        vector = outputs.cpu().numpy()
        vector = vector.reshape(-1)

        return vector
    
if __name__ == "__main__":
    mulan = MuLan()
    embedding1 = mulan.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")
    # embedding2 = mulan.get_embedding("Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")
    
    embedding2 = mulan.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")
    

    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)