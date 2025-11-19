import pickle
import random
import os
import time
import torch
import random
import yaml
import json

import numpy as np
from model.mamba import TransformerXL
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_sample', type=int, default=1, help='number of samples to generate')
    parser.add_argument('-o', '--output_prefix', type=str, default='output/1', help='output midi prefix')
    parser.add_argument('-t', '--temperature', type=float, default=1.2, help='temperature for sampling')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='top_p for nucleus sampling')
    args = parser.parse_args()

    cfg = yaml.full_load(open("./config/config_mamba.yml", 'r')) 
    inferenceConfig = cfg['INFERENCE']
    
    os.environ['CUDA_VISIBLE_DEVICES'] = inferenceConfig['gpuID']

    print('='*2, 'Inferenc configs', '='*5)
    print(json.dumps(inferenceConfig, indent=1, sort_keys=True))

    # checkpoint information
    CHECKPOINT_FOLDER = inferenceConfig['experiment_dir']
    midi_folder = inferenceConfig["generated_dir"]
    additional_midi_dirs = inferenceConfig.get("additional_midi_dirs") or []
    if isinstance(additional_midi_dirs, str):
        additional_midi_dirs = [additional_midi_dirs]

    checkpoint_type = inferenceConfig['checkpoint_type']
    
    
    model_path = './ckpt/ep_300.pth.tar'
    output_prefix = 'output/t1/remi_mb/t' + str(args.temperature) + 'p' + str(args.top_p)
    os.makedirs(output_prefix, exist_ok=True)

    pretrainCfg = yaml.full_load(open("./config/config_mamba.yml", 'r'))
    modelConfig = pretrainCfg['MODEL']



    # load dictionary
    event2word, word2event = pickle.load(open(inferenceConfig['dictionary_path'], 'rb'))

    # declare model
    device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model = TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)

    # inference
    song_time_list = []
    words_len_list = []
    num_samples = args.num_sample
    for idx in range(num_samples):
        print(f'==={idx}/{num_samples}===')
        print(midi_folder, output_prefix + str(idx))
        extra_paths = [
            os.path.join(extra_dir, f"{output_prefix}{idx}.mid")
            for extra_dir in additional_midi_dirs if extra_dir
        ]
        song_time, word_len = model.inference(
            model_path = model_path,
            token_lim=7680,
            strategies=['temperature', 'nucleus'],
            params={'t': args.temperature, 'p': args.top_p},
            bpm=120,
            output_path=f'{output_prefix}/{idx}_t{args.temperature}p{args.top_p}.mid',
            extra_output_paths=extra_paths or None)

        print('song time:',  song_time)
        print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
    

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }
    

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)

if __name__ == '__main__':
    main()
