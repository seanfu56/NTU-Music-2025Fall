# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F
import librosa
from sklearn.metrics.pairwise import cosine_similarity


class MERT:

    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        
        self.sample_rate = self.processor.sampling_rate

        self.model.eval()

    def get_embedding(self, audio_path):
        audio = librosa.load(audio_path, sr=self.sample_rate)[0]
        inputs = self.processor(audio, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        vector = time_reduced_hidden_states.cpu().numpy()
        vector = vector.reshape(-1)
        

        return vector

if __name__ == "__main__":
    mert = MERT()
    embedding1 = mert.get_embedding("Deep_MIR_hw2/target_music_list_60s/4_jazz_120_beat_3-4.wav")
    embedding2 = mert.get_embedding("Deep_MIR_hw2/target_music_list_60s/\u7af9\u7b1b\uff5c\u8fd9\u4e16\u754c\u90a3\u4e48\u591a\u4eba_cover \u83ab\u6587\u851a_60s.mp3")
    

    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    print("Cosine Similarity:", sim)