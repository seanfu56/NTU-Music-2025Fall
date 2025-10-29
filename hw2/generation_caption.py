import random
import glob
import json

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def main():
    
    SEED = 42
    set_seed(SEED)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_cfg
    ).to("cuda")
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    prompt = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": ""},
            {"type": "text", "text": "5–8 audio-only sentences: instruments+melody, genre/mood, ~BPM/meter & rhythm, key/mode+chords, form, timbre/production, dynamics/articulation, techniques. Include mm:ss timestamps; use 'unknown' if unsure"},
        ]
    }]
    
    target_music_dir = 'Deep_MIR_hw2/target_music_list_60s'
    target_music_paths = sorted(glob.glob(f'{target_music_dir}/*'))
    
    reference_music_dir = 'Deep_MIR_hw2/reference_music_list_60s'
    reference_music_paths = sorted(glob.glob(f'{reference_music_dir}/*'))
    
    target_caption_results = {}
    reference_caption_results = {}
    
    for path in target_music_paths:
        
        audio, sr = librosa.load(path, sr=processor.feature_extractor.sampling_rate)
        audios = [audio]
        
        text = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = processor(
            text=text,
            audios=audios,
            return_tensors="pt"
        ).to("cuda")
        
        generated_ids = model.generate(
            **inputs,
            max_length=512,
            use_cache=True
        )

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        target_caption_results[path] = response

    for path in reference_music_paths:
        
        audio, sr = librosa.load(path, sr=processor.feature_extractor.sampling_rate)
        audios = [audio]
        
        text = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = processor(
            text=text,
            audios=audios,
            return_tensors="pt"
        ).to("cuda")
        
        generated_ids = model.generate(
            **inputs,
            max_length=512,
            use_cache=True
        )

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        reference_caption_results[path] = response
        
    model_name_sanitized = model_name.split("/")[-1]
    
    with open(f'output/generation/target_caption_results_{model_name_sanitized}.json', 'w') as f:
        json.dump(target_caption_results, f, indent=4)

    with open(f'output/generation/reference_caption_results_{model_name_sanitized}.json', 'w') as f:
        json.dump(reference_caption_results, f, indent=4)
        
if __name__ == "__main__":
    main()