import os, glob, inspect, traceback
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

import encoders

import warnings
warnings.filterwarnings("ignore")

import json

def iter_encoders():
    
    names = getattr(encoders, "__all__", [])
    
    for name in names:
        
        def make(name=name, enc_mod=encoders):
            cls = getattr(enc_mod, name)   # 觸發 __getattr__ 懶載入
            return cls()

        yield name, make

    # print(names)
    
def collect_paths(folder, patterns=("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(paths)

    
def main():
    
    TARGET_DIR = "Deep_MIR_hw2/target_music_list_60s"
    REF_DIR    = "Deep_MIR_hw2/reference_music_list_60s"
    
    target_paths = collect_paths(TARGET_DIR)
    ref_paths    = collect_paths(REF_DIR)
    
    encoder_list = list(iter_encoders())
    # encoder_list = encoder_list[2:]

    for name, make in encoder_list:
        
        encoder = make()
        
        target_embeddings = []
        for target_path in target_paths:
            embedding = encoder.get_embedding(target_path)
            target_embeddings.append(embedding)
        target_embeddings = np.array(target_embeddings)
        
        ref_embeddings = []
        for ref_path in ref_paths:
            embedding = encoder.get_embedding(ref_path)
            ref_embeddings.append(embedding)
            # print(embedding.shape)
        ref_embeddings = np.array(ref_embeddings)
        
        pair_list = []

        for i, target_path in enumerate(target_paths):
            target_emb = target_embeddings[i:i+1]
            sims = cosine_similarity(target_emb, ref_embeddings)[0]
            ranked_indices = np.argsort(sims)[::-1]
            top1_index = ranked_indices[0]
            top1_ref_path = ref_paths[top1_index]
            print(f"Encoder: {name}, Target: {os.path.basename(target_path)}, Top-1 Ref: {os.path.basename(top1_ref_path)}, Similarity: {sims[top1_index]:.4f}")

            pair_list.append( (target_path, top1_ref_path, float(sims[top1_index])) )

        with open(f"output/retrieval/results_{name}.json", "w") as f:
            json.dump(pair_list, f, indent=2)

if __name__ == "__main__":
    main()