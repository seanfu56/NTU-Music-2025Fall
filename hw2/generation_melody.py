import os
import glob
import generative_models
import json
import soundfile as sf
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def iter_generators():
    names = getattr(generative_models, "__all_con__", [])
    
    for name in names:
        
        def make(name=name, gen_mod=generative_models):
            cls = getattr(gen_mod, name)   # 觸發 __getattr__ 懶載入
            return cls()

        yield name, make
        
def collect_paths(folder, patterns=("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(paths)    


def main():
    
    set_seed(42)
    
    with open('output/generation/target_caption_results_Qwen2-Audio-7B-Instruct.json', 'r') as f:
        caption_results = json.load(f)
    
    generator_list = list(iter_generators())
    
    for name, make in generator_list:
        
        generator = make()
        
        output_dir = f"output/generation/{name}"
        os.makedirs(output_dir, exist_ok=True)
        
        for key, value in caption_results.items():
            base_name = os.path.basename(key)
            output_path = os.path.join(output_dir, base_name)

            music, sr = generator.generate(
                prompt = value,
                audio_path = key
            )
            # print(music.shape)
            print(f"Generated {output_path} using {name}")
            
            wav = music

            sf.write(output_path, wav, samplerate=sr)

if __name__ == "__main__":
    main()