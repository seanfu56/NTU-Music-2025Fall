import glob
import json

from benchmark import clap, aesthetics, melody

def main():


    aesthetics_model = aesthetics.Aesthetics()
    
    
    target_paths = glob.glob('Deep_MIR_hw2/target_music_list_60s/*')
    
    result_jsons = []
    
    for target_path in target_paths:
        
        ce, cu, pc, pq = aesthetics_model.get_score(target_path)
        
        result_jsons.append({
            "target_path": target_path,
            "aesthetics_CE": float(ce),
            "aesthetics_CU": float(cu),
            "aesthetics_PC": float(pc),
            "aesthetics_PQ": float(pq),
        })

    with open(f'output/retrieval/target.json', 'w') as f:
        json.dump(result_jsons, f, indent=4)
        
        
if __name__ == "__main__":
    main()