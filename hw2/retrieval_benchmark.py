import glob
import json

from benchmark import clap, aesthetics, melody
import encoders

def main():

    encoder_list = getattr(encoders, "__all__", [])

    clap_model = clap.CLAP()
    aesthetics_model = aesthetics.Aesthetics()
    
    for encoder in encoder_list:
        
        result_jsons = []
        
        json_file = f'output/retrieval/results_{encoder}.json'
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for item in data:
            target_path, ref_path, sim = item
            
            melody_sim = melody.melody_similarity(target_path, ref_path)
            ce, cu, pc, pq = aesthetics_model.get_score(ref_path)
            clap_sim = clap_model.get_similarity(target_path, ref_path)
            
            result_jsons.append({
                "encoder": encoder,
                "target_path": target_path,
                "ref_path": ref_path,
                "encoder_similarity": float(sim),
                "melody_similarity": float(melody_sim),
                "aesthetics_CE": float(ce),
                "aesthetics_CU": float(cu),
                "aesthetics_PC": float(pc),
                "aesthetics_PQ": float(pq),
                "clap_similarity": float(clap_sim)
            })

        with open(f'output/retrieval/benchmark_{encoder}.json', 'w') as f:
            json.dump(result_jsons, f, indent=4)
            
            
if __name__ == "__main__":
    main()