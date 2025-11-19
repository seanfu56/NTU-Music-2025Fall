import sys
import os
import argparse

import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import logging
import wandb

import os
import torch
from mamba_ssm import Mamba

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch device_count  =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"logical cuda:{i} -> {p.name}, {p.total_memory/1024**3:.1f} GB")

class Saver(object):
    def __init__(
            self, 
            exp_dir, 
            mode='w'):

        self.exp_dir = exp_dir
        self.init_time = time.time()
        self.global_step = 0

        # makedirs
        os.makedirs(exp_dir, exist_ok=True)

        # logging config
        path_logger = os.path.join(exp_dir, 'log.txt')
        logging.basicConfig(
                level=logging.DEBUG,
                format='%(message)s',
                filename=path_logger,
                filemode=mode)
        self.logger = logging.getLogger('training monitor')

    def add_summary_msg(self, msg):
        self.logger.debug(msg)

    def add_summary(
            self, 
            key, 
            val, 
            step=None, 
            cur_time=None):

        if cur_time is None:
            cur_time = time.time() - self.init_time
        if step is None:
            step = self.global_step

        # write msg (key, val, step, time)
        if isinstance(val, float):
            msg_str = '{:10s} | {:.10f} | {:10d} | {}'.format(
                    key, 
                    val, 
                    step, 
                    cur_time
                )
        else:
            msg_str = '{:10s} | {} | {:10d} | {}'.format(
                    key, 
                    val, 
                    step, 
                    cur_time
                )

        self.logger.debug(msg_str)
    
    def save_model(
            self, 
            model, 
            optimizer=None, 
            outdir=None, 
            name='model'):

        if outdir is None:
            outdir = self.exp_dir
        print(' [*] saving model to {}, name: {}'.format(outdir, name))
        # torch.save(model, os.path.join(outdir, name+'.pt'))
        torch.save(model.state_dict(), os.path.join(outdir, name+'_params.pt'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(outdir, name+'_opt.pt'))
            
    def load_model(
            self, 
            path_exp, 
            device='cpu', 
            name='model.pt'):

        path_pt = os.path.join(path_exp, name)
        print(' [*] restoring model from', path_pt)
        model = torch.load(path_pt, map_location=torch.device(device))
        return model
        
    def global_step_increment(self):
        self.global_step += 1


################################################################################
# config
################################################################################

MODE = 'inference' 
# MODE = 'inference' 

###--- data ---###
path_data_root = './Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')

###--- training config ---###
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8    
path_exp = 'exp'
batch_size = 4
gid = 0
init_lr = 0.0001

###--- fine-tuning & inference config ---###
# info_load_model = (
#     # path to ckpt for loading
#     '/volume/ai-music-wayne/aaai/from-scratch/cp-linear/exp_base_fs',
#     # loss
#     29                               
# )
info_load_model = None
path_gendir = 'gen_midis_mamba'
num_songs = 1

################################################################################
# File IO
################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4


def write_midi(words, path_outfile, word2event):
    
    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]
                
                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)
                
                all_notes.append(
                    Note(
                        pitch=int(pitch), 
                        start=cur_pos, 
                        end=end, 
                        velocity=int(velocity))
                    )
            except:
                continue
        else:
            pass
    
    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)
    
    with open(path_outfile.replace('.mid', '.json'), 'w') as f:
        # output words
        json.dump(words.tolist(), f)


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)
    
    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################

def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual + feed-forward."""

    def __init__(self, d_model, d_state, d_conv, expand, d_inner, dropout):
        super().__init__()
        self.mixer_norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model)
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(self.mixer_norm(x))
        x = self.mamba_dropout(x)
        x = x + residual

        residual = x
        x = self.ff(self.ff_norm(x))
        x = self.ff_dropout(x)
        x = x + residual
        return x


class MambaModel(nn.Module):
    """
    用 Mamba 取代原本的 TransformerEncoder / RecurrentEncoder，
    其他介面（train_step, inference_from_scratch 等）儘量保持一樣。
    """
    def __init__(self, n_token, is_training=True):
        super(MambaModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token   # list of 7 vocab sizes
        self.d_model = D_MODEL
        self.n_layer = N_LAYER
        self.dropout = 0.1
        self.n_head = N_HEAD
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # tempo, chord, barbeat, type, pitch, duration, velocity
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]

        # --- modules config --- #
        # embeddings
        print('>>>>> (MambaModel) n_token:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type      = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # 將各個 embedding concat 起來再映射到 d_model
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        # === Mamba stack（取代原本 transformer_encoder / recurrent encoder） ===
        # 這裡簡單做一個 n_layer 層的 Mamba 堆疊再接一個 LayerNorm
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                d_inner=self.d_inner,
                dropout=self.dropout
            )
            for _ in range(self.n_layer)
        ])
        self.mamba_norm = nn.LayerNorm(self.d_model)
        self.mamba_dropout = nn.Dropout(self.dropout)

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        # individual output heads
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])

    # ======== utils ========

    def compute_loss(self, predict, target, loss_mask):
        # predict: (B, C, L), target: (B, L), loss_mask: (B, L)
        loss = self.loss_func(predict, target)  # (B, L)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    # ======== forward (training) ========

    def train_step(self, x, target, loss_mask):
        """
        x: (B, L, 7)
        target: (B, L, 7)
        loss_mask: (B, L)
        """
        h, y_type  = self.forward_hidden(x)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = \
            self.forward_output(h, target)
         
        # reshape (B, L, F) -> (B, F, L) for CrossEntropyLoss
        y_tempo     = y_tempo.permute(0, 2, 1)
        y_chord     = y_chord.permute(0, 2, 1)
        y_barbeat   = y_barbeat.permute(0, 2, 1)
        y_type      = y_type.permute(0, 2, 1)
        y_pitch     = y_pitch.permute(0, 2, 1)
        y_duration  = y_duration.permute(0, 2, 1)
        y_velocity  = y_velocity.permute(0, 2, 1)
        
        # loss
        loss_tempo = self.compute_loss(y_tempo,    target[..., 0], loss_mask)
        loss_chord = self.compute_loss(y_chord,    target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(y_barbeat, target[..., 2], loss_mask)
        loss_type  = self.compute_loss(y_type,     target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(y_pitch,    target[..., 4], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 6], loss_mask)

        return (loss_tempo, loss_chord, loss_barbeat,
                loss_type, loss_pitch, loss_duration, loss_velocity)

    def _encode_with_mamba(self, x):
        """
        x: (B, L, D_MODEL)
        回傳: (B, L, D_MODEL)
        """
        h = x
        for layer in self.mamba_layers:
            h = layer(h)
        h = self.mamba_norm(h)
        h = self.mamba_dropout(h)
        return h

    def forward_hidden(self, x, memory=None, is_training=True):
        """
        linear transformer 版改成 Mamba 版：
        x: (B, L, 7)
        訓練時 B=batch, L=seq_len
        推論時（你的 inference_from_scratch），B=1, L=1
        """

        # embeddings
        emb_tempo    = self.word_emb_tempo(x[..., 0])
        emb_chord    = self.word_emb_chord(x[..., 1])
        emb_barbeat  = self.word_emb_barbeat(x[..., 2])
        emb_type     = self.word_emb_type(x[..., 3])
        emb_pitch    = self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ],
            dim=-1
        )  # (B, L, sum_emb_sizes)

        emb_linear = self.in_linear(embs)        # (B, L, D_MODEL)
        pos_emb    = self.pos_emb(emb_linear)    # (B, L, D_MODEL)

        # Mamba encoding
        h = self._encode_with_mamba(pos_emb)     # (B, L, D_MODEL)

        # project type
        y_type = self.proj_type(h)               # (B, L, n_type)

        # 為了相容原本介面，推論也回傳 memory，但這裡不使用 memory
        if is_training:
            return h, y_type
        else:
            return h, y_type, None

    def forward_output(self, h, y):
        """
        for training
        h: (B, L, D_MODEL)
        y: (B, L, 7)  (teacher forcing)
        """
        tf_skip_type = self.word_emb_type(y[..., 3])  # (B, L, 32)

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)  # (B, L, D+32)
        y_  = self.project_concat_type(y_concat_type)

        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        return  y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity

    def forward_output_sampling(self, h, y_type, temperature, top_p):
        """
        for inference
        h:      (1, 1, D_MODEL)
        y_type: (1, 1, n_type)
        """
        # sample type
        y_type_logit = y_type[0, 0, :]  # (n_type,)
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])
        ).long().cuda().unsqueeze(0)  # (1, 1)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)  # (1, 32)

        # concat
        y_concat_type = torch.cat([h.squeeze(1), tf_skip_type], dim=-1)  # (1, D+32)
        y_  = self.project_concat_type(y_concat_type)  # (1, D_MODEL)

        # project other
        y_tempo    = self.proj_tempo(y_)     # (1, n_tempo)
        y_chord    = self.proj_chord(y_)     # (1, n_chord)
        y_barbeat  = self.proj_barbeat(y_)   # (1, n_barbeat)
        y_pitch    = self.proj_pitch(y_)     # (1, n_pitch)
        y_duration = self.proj_duration(y_)  # (1, n_duration)
        y_velocity = self.proj_velocity(y_)  # (1, n_velocity)
        
        # sampling gen_cond
        cur_word_tempo    = sampling(y_tempo,    t=temperature, p=top_p)
        cur_word_barbeat  = sampling(y_barbeat, t=temperature, p=top_p)
        cur_word_chord    = sampling(y_chord,   t=temperature, p=top_p)
        cur_word_pitch    = sampling(y_pitch,   t=temperature, p=top_p)
        cur_word_duration = sampling(y_duration, t=temperature, p=top_p)
        cur_word_velocity = sampling(y_velocity, t=temperature, p=top_p)

        # collect
        next_arr = np.array([
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_type,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
        ])        
        return next_arr

    def inference_from_scratch(self, dictionary, temperature, top_p):
        """使用整段前綴重算隱狀態，確保每一步都看到完整上下文。"""
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([
            [0, 0, 1, 1, 0, 0, 0], # bar
        ])

        with torch.no_grad():
            final_res = [row.copy() for row in init]
            cnt_bar = 1

            print('------ initiate ------')
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])

            print('------ generate ------')
            while True:
                prefix = np.stack(final_res, axis=0)
                input_ = torch.from_numpy(prefix).long().cuda().unsqueeze(0)
                h, y_type = self.forward_hidden(input_, is_training=True)
                h_last = h[:, -1:, :]
                y_type_last = y_type[:, -1:, :]

                next_arr = self.forward_output_sampling(h_last, y_type_last, temperature, top_p)
                final_res.append(next_arr)
                print('bar:', cnt_bar, end='  ==')
                print_word_cp(next_arr)

                if word2event['type'][next_arr[3]] == 'EOS':
                    break
                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1
                    
                if cnt_bar >= 32:
                    break

        print('\n--------[Done]--------')
        final_res = np.stack(final_res, axis=0)
        print(final_res.shape)
        return final_res
    
    def inference_condition(self, dictionary, numpy_file, temperature, top_p):
        """使用整段前綴重算隱狀態，確保每一步都看到完整上下文。"""
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        # init = np.array([
        #     [0, 0, 1, 1, 0, 0, 0], # bar
        # ])
        
        init = np.load(numpy_file)

        with torch.no_grad():
            final_res = [row.copy() for row in init]
            cnt_bar = 1

            print('------ initiate ------')
            # for step in range(init.shape[0]):
                # print_word_cp(init[step, :])

            print('------ generate ------')
            while True:
                prefix = np.stack(final_res, axis=0)
                input_ = torch.from_numpy(prefix).long().cuda().unsqueeze(0)
                h, y_type = self.forward_hidden(input_, is_training=True)
                h_last = h[:, -1:, :]
                y_type_last = y_type[:, -1:, :]

                next_arr = self.forward_output_sampling(h_last, y_type_last, temperature, top_p)
                final_res.append(next_arr)
                print('bar:', cnt_bar, end='  ==')
                # print_word_cp(next_arr)

                if word2event['type'][next_arr[3]] == 'EOS':
                    break
                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1
                    
                if cnt_bar >= 32:
                    break

        print('\n--------[Done]--------')
        final_res = np.stack(final_res, axis=0)
        print(final_res.shape)
        return final_res
    
    
def train():
    # hyper params
    n_epoch = 300
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)

    # create saver
    saver_agent = Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)
   
    # init
    net = MambaModel(n_class)
    net.cuda()
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    # load model
    if info_load_model:
        path_ckpt = info_load_model[0] # path to ckpt dir
        loss = info_load_model[1] # loss
        name = 'loss_' + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
        print('[*] load model from:',  path_saved_ckpt)
        net.load_state_dict(torch.load(path_saved_ckpt))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack
    train_x = train_data['x']
    train_y = train_data['y']
    train_mask = train_data['mask']
    num_batch = len(train_x) // batch_size
    
    print('     num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)
    print('    train_mask:', train_mask.shape)

    # run
    
    wandb.init(project="music-hw3-cp", name="transformer_xl_cpword")
    
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(7)
        
        st_time = time.time()

        for bidx in range(num_batch): # num_batch 
            saver_agent.global_step_increment()

            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).long().cuda()
            batch_y = torch.from_numpy(batch_y).long().cuda()
            batch_mask = torch.from_numpy(batch_mask).float().cuda()

            # run
            losses = net.train_step(batch_x, batch_y, batch_mask)
            loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7

            # Update
            net.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6]))
            sys.stdout.flush()

            # acc
            acc_losses += np.array([l.item() for l in losses])
            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())
        
        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
              acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3], acc_losses[4], acc_losses[5], acc_losses[6])
        print('    >', each_loss_str)

        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)

        # save model, with policy
        loss = epoch_loss
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='mb_loss_' + str(fn))
        elif 0.05 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='mb_loss_' + str(fn))
        elif loss <= 0.05:
            print('Finished')
            return  
        else:
            saver_agent.save_model(net, name='mb_loss_high')
            
        wandb.log({'train/loss': epoch_loss, 'epoch': epoch+1, 'time/epoch': time.time()-st_time}) 


def generate(
        use_condition,
        path_condition_file,
        num_songs,
        path_gendir,
        temperature,
        top_p
):
    # path
    # path_ckpt = info_load_model[0] # path to ckpt dir
    # loss = info_load_model[1] # loss
    # name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join('ckpt/mb_loss_8_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = MambaModel(n_class, is_training=False)
    net.cuda()
    net.eval()
    
    # load model
    print('[*] load model from:',  path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0 
    sidx = 0
    while sidx < num_songs:
        try:
            start_time = time.time()
            print('current idx:', sidx)
            path_outfile = os.path.join(path_gendir, f'{sidx}_t{temperature}_p{top_p}.mid')

            if use_condition:
                res = net.inference_condition(dictionary, path_condition_file, temperature, top_p)
            else:
                res = net.inference_from_scratch(dictionary, temperature, top_p)
            write_midi(res, path_outfile, word2event)

            song_time = time.time() - start_time
            word_len = len(res)
            print('song time:', song_time)
            print('word_len:', word_len)
            words_len_list.append(word_len)
            song_time_list.append(song_time)

            sidx += 1
        except KeyboardInterrupt:
            raise ValueError(' [x] terminated.')
        except Exception as e:
            print(' [x] error:', e)
            continue
  
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
    # -- training -- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', action='store_true', help='use condition generation')
    parser.add_argument('--cond_file', type=str, default='', help='path to condition numpy file')
    parser.add_argument('-n', '--num_songs', type=int, default=50, help='number of songs to generate')
    parser.add_argument('-o', '--output_dir', type=str, default='gen_midis', help='output directory for generated midis')
    parser.add_argument('-t', '--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help='nucleus sampling top p value')
    args = parser.parse_args()
    
    if MODE == 'train':
        train()

    # -- inference -- #
    elif MODE == 'inference':
        generate(
            args.cond,
            args.cond_file,
            args.num_songs,
            args.output_dir,
            args.temperature,
            args.top_p
        )
        
    else:
        pass
