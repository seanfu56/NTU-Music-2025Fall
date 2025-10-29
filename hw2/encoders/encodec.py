import torch
import librosa
from transformers import EncodecModel, AutoProcessor

class Encodec:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.sample_rate = self.processor.sampling_rate
        
    def get_embedding(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=self.sample_rate).to(self.device)
        with torch.no_grad():
            outputs = self.model.encode(**inputs).audio_codes
        
        # print(outputs)
        print(outputs.shape)
        z = outputs.squeeze().permute(1, 0)  # (num_frames, codebook_size) -> (codebook_size, num_frames)
        vector = z.cpu().numpy()
        return vector
    
if __name__ == "__main__":
    encodec = Encodec()
    embedding1 = encodec.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")

    embedding2 = encodec.get_embedding("Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")


    # embedding2 = encodec.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")
    
    
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)