import glob
import json
import os

from benchmark import clap, aesthetics, melody
import generative_models

def main():

    encoder_list = getattr(generative_models, "__all__", [])

    clap_model = clap.CLAP()
    aesthetics_model = aesthetics.Aesthetics()
    
    target_paths = glob.glob('Deep_MIR_hw2/target_music_list_60s/*')
    
    for encoder in encoder_list:
        
        result_json = []
        
        for target_path in target_paths:
        
            generation_path = f'output/generation/{encoder}/{os.path.basename(target_path)}'
            
            melody_sim = melody.melody_similarity(target_path, generation_path)
            ce, cu, pc, pq = aesthetics_model.get_score(generation_path)
            clap_sim = clap_model.get_similarity(target_path, generation_path)
            
            result_json.append({
                "encoder": encoder,
                "target_path": target_path,
                "generation_path": generation_path,
                "melody_similarity": float(melody_sim),
                "aesthetics_CE": float(ce),
                "aesthetics_CU": float(cu),
                "aesthetics_PC": float(pc),
                "aesthetics_PQ": float(pq),
                "clap_similarity": float(clap_sim)
            })

        with open(f'output/generation/benchmark_{encoder}.json', 'w') as f:
            json.dump(result_json, f, indent=4)

if __name__ == "__main__":
    main()