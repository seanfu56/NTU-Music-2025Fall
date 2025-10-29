import numpy as np
import librosa
from music2latent import EncoderDecoder

class M2L:
    def __init__(self):
        self.encdec = EncoderDecoder()

    def get_embedding(self, audio_path, sr=44100, seconds=60):
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if seconds is not None:
            L = sr * seconds
            if len(y) >= L:
                y = y[:L]
            else:
                y = np.pad(y, (0, L - len(y)))
        z = self.encdec.encode(y)        # 期望 shape: [1, 64, T]
        if hasattr(z, "detach"):    # torch.Tensor -> numpy
            z = z.detach().cpu().numpy()
        z = z[0]                    # [64, T]
        # 先把每一幀做 L2 normalize（可選），再沿時間做平均
        z = z / (np.linalg.norm(z, axis=0, keepdims=True) + 1e-8)
        emb = z.mean(axis=-1)       # [64]
        # 全局 L2 normalize，便於直接用點積當 cosine
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    
if __name__ == "__main__":
    m2l = M2L()
    
    embedding1 = m2l.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")
    # embedding2 = m2l.get_embedding("Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")

    embedding2 = m2l.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")

    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)