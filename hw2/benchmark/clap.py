import librosa
import torch
from transformers import ClapProcessor, ClapModel

class CLAP:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        self.model.eval()

    def get_embedding(self, audio_path):
        audio = librosa.load(audio_path, sr=self.sample_rate)[0]
        inputs = self.processor(audios=audio, return_tensors="pt", sampling_rate=self.sample_rate).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)
        
        vector = outputs.cpu().numpy()
        vector = vector.reshape(-1)

        return vector
    
    def get_similarity(self, audio_path1, audio_path2):
        embedding1 = self.get_embedding(audio_path1)
        embedding2 = self.get_embedding(audio_path2)

        from sklearn.metrics.pairwise import cosine_similarity

        sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

        return sim[0][0]
    
if __name__ == "__main__":
    clap = CLAP()
    similarity = clap.get_similarity("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav", "Deep_MIR_hw2/target_music_list_60s/6_rock_102_beat_3-4.wav")

    print("Cosine Similarity:", similarity)
