import sys
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import miditoolkit
import shutil
import copy
import os
import time
import json
from sklearn.model_selection import train_test_split
# from modules import MemTransformerLM
from glob import glob

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import collections
import pickle 
import numpy as np

# import saver

# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}

import os
import time
import torch
import logging
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt

import wandb


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
        torch.save(model, os.path.join(outdir, name+'.pt'))
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

"""
file modes
'a': 
    Opens a file for appending. The file pointer is at the end of the file if the file exists. 
    That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.

'w':
    Opens a file for writing only. Overwrites the file if the file exists.
    If the file does not exist, creates a new file for writing.
"""

def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=100):

    # load logfile
    monitor_vals = collections.defaultdict(list)
    with open(path_logfile, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    # collect
    step_train = [item[1] for item in monitor_vals['train loss']]
    vals_train = [item[0] for item in monitor_vals['train loss']]

    step_valid = [item[1] for item in monitor_vals['valid loss']]
    vals_valid = [item[0] for item in monitor_vals['valid loss']]

    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure)

'''
author: wayn391@mastertones
'''

import os
import time
import torch
import logging
import datetime
import collections
import numpy as np
import matplotlib.pyplot as plt


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

"""
file modes
'a': 
    Opens a file for appending. The file pointer is at the end of the file if the file exists. 
    That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.

'w':
    Opens a file for writing only. Overwrites the file if the file exists.
    If the file does not exist, creates a new file for writing.
"""

def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=100):

    # load logfile
    monitor_vals = collections.defaultdict(list)
    with open(path_logfile, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    # collect
    step_train = [item[1] for item in monitor_vals['train loss']]
    vals_train = [item[0] for item in monitor_vals['train loss']]

    step_valid = [item[1] for item in monitor_vals['valid loss']]
    vals_valid = [item[0] for item in monitor_vals['valid loss']]

    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure)


def wrtie_midi(words, path_midi, word2event, extra_paths=None):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile()
    cur_pos = 0
    
    for i in range(len(events)-3):
        cur_event = events[i]
        # print(cur_event)
        name = cur_event.split('_')[0]
        attr = cur_event.split('_')
        if name == 'Bar':
            bar_cnt += 1
        elif name == 'Beat':
            cur_beat = int(attr[1])
            cur_pos = bar_cnt * BAR_RESOL + cur_beat * TICK_RESOL
        elif name == 'Chord':
            chord_text = attr[1] + '_' + attr[2]
            midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
        elif name == 'Tempo':
            midi_obj.tempo_changes.append(
                TempoChange(tempo=int(attr[1]), time=cur_pos))
        else:
            if 'Note_Pitch' in events[i] and \
            'Note_Velocity' in events[i+1] and \
            'Note_Duration' in events[i+2]:

                pitch = int(events[i].split('_')[-1])
                duration = int(events[i+2].split('_')[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration 
                velocity = int(events[i+1].split('_')[-1])
                notes_all.append(
                    Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))
                
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)

    with open(path_midi.replace('.mid', '.json'), 'w') as f:
        # output words
        json.dump([int(word) for word in words], f)


# ================================ #
def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

import sys
import math
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            # print("w",w.shape)
            # print("mems",mems.shape)
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias
        AC = rw_head_q.permute(1, 2, 0, 3) @ w_head_k.permute(1, 2, 3, 0)

        rr_head_q = w_head_q + r_r_bias
        BD = rr_head_q.permute(1, 2, 0, 3) @ r_head_k.permute(1, 2, 0)
        BD = F.pad(BD, [1, 0]).view(BD.size(0), BD.size(
            1), BD.size(3) + 1, BD.size(2))[:, :, 1:].view_as(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask.permute(2, 0, 1)[:, None, :, :], -float('inf')).type_as(attn_score)

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = attn_prob @ w_head_v.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)



class MemTransformerLM(nn.Module):
    def __init__(self, modelConfig,
                tie_projs=[False], cutoffs=[], 
                 is_training=True):
        super(MemTransformerLM, self).__init__()

        self.n_token = modelConfig['n_token']
        self.n_layer= modelConfig['n_layer']
        self.n_head= modelConfig['n_head']
        self.d_model = modelConfig['d_model']
        self.d_embed = d_model if modelConfig['d_embed'] is None else modelConfig['d_embed']
        self.d_head = self.d_model // self.n_head
        self.d_inner= modelConfig['d_inner']

        self.mem_len = modelConfig['mem_len']
        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len  #70+0+512

        self.dropout= modelConfig['dropout']
        self.dropatt = modelConfig['dropatt']

        self.clamp_len = modelConfig['clamp_len']
        self.div_val = modelConfig['div_val']

        #choice
        self.pre_lnorm = modelConfig['pre_lnorm']
        self.same_length = modelConfig['same_length']
        self.is_training = is_training

        #building layers
        self.drop = nn.Dropout(self.dropout)
        self.word_emb = Embeddings(self.n_token, self.d_model)

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                    tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                    dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
            )

        # output layer
        self.linear_proj = nn.Linear(self.d_model, self.n_token)

        # loss
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self._create_params()

    def compute_loss(self, predict, target, loss_mask=None):
        '''
        predict, target,
        input:  (N, C, ...)
        target: (N, ...)
        '''
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        
        if mems is None: return None
        # mems is not None
        # assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems



    def _forward(self, dec_inp, mems=None):
        '''
        output of _forward: step x batch x n_feat
        predict = self.linear_proj(hidden)
        '''

        qlen, bsz = dec_inp.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        word_emb = self.word_emb(dec_inp)

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len

            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]


        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        hids.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def generate(self, data, *mems):
        if not mems: mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)
        predict = self.linear_proj(hidden[-1:])
        return predict, new_mems

    def forward(self, data, target, mask, *mems):
        if not mems: mems = self.init_mems()
        
        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        predict = self.linear_proj(pred_hid)

        predict = predict.permute(1, 2, 0)
        target = target.permute(1, 0)

        loss = self.compute_loss(predict, target, mask)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

class TransformerXL(object):
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings    
        self.n_layer= modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len= modelConfig['seq_len']
        self.mem_len =  modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']

        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']

        #mode
        self.is_training = is_training
        self.device = device  
        

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
            
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)


    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.modelConfig, is_training=self.is_training)

        st_eopch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model, map_location='cuda:0')
            print('Pretrained model config:')
            print('epoch: ', checkpoint['epoch'])
            print('best_loss: ', checkpoint['best_loss'])
            print(json.dumps(checkpoint['model_setting'], indent=1, sort_keys=True))
            print(json.dumps(checkpoint['train_setting'], indent=1, sort_keys=True))

            try:
                model.load_state_dict(checkpoint['state_dict'])
                print('{} loaded.'.format(pretrain_model))  
            except:
                print('Loaded weights have different shapes with the model. Please check your model setting.')
                exit()
            st_eopch = checkpoint['epoch'] + 1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) 
        return st_eopch ,model.to(self.device)


    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root,'ep_{}.pth.tar'.format(state['epoch'])))

    def train_loss_record(self, epoch, train_loss,checkpoint_dir, val_loss=None):

        if val_loss:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss],
                    'val_loss': ['%.3f'%val_loss]})
            
        else:
            df = pd.DataFrame({'epoch': [epoch+1],
                    'train_loss': ['%.3f'%train_loss]})

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(os.path.join(checkpoint_dir, 'loss.csv'), mode='a', header=False,  index=False)

    def train(self, train_data, trainConfig, device, resume):
        checkpoint_dir = trainConfig['experiment_Dir']
        batch_size = trainConfig['batch_size']
        data_ROOT = trainConfig['ROOT']
        torch.manual_seed(trainConfig["seed"])

        # create saver
        saver_agent = Saver(checkpoint_dir)

        wandb_config = {
            'model_config': self.modelConfig,
            'train_config': trainConfig
        }
        
        wandb.init(
            project="music-hw3-cp",
            name="transformer_xl_remi",
            config=wandb_config
        )

        #Prepare model
        if resume != 'None':
            st_epoch, model = self.get_model(resume)
            print('Continue to train from {} epoch'.format(st_epoch))
        else:
            st_epoch, model = self.get_model()

        optimizer = optim.Adam(model.parameters(), lr=trainConfig['lr'])
        train_step = 0
        epoch_train_loss = []
        save_freq = trainConfig['save_freq']
        
        n_parameters = network_paras(model)
        print('n_parameters: {:,}'.format(n_parameters))
        saver_agent.add_summary_msg(
            ' > params amount: {:,d}'.format(n_parameters))

        # unpack
        train_x = train_data['x'] 
        train_y = train_data['y'] 
        mask = train_data['mask'] 
        num_groups = train_data['num_groups'] 

        num_batches = len(train_x ) // batch_size
        
        print('>>> Start training')
        for epoch in range(st_epoch, trainConfig['num_epochs']):
            saver_agent.global_step_increment()

            train_loss = []
            st_time = time.time()
            model.train()

            for bidx in range(num_batches):
                
                model.zero_grad()

                # index
                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                # get batch
                batch_x = train_x[bidx_st:bidx_ed]
                batch_y = train_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                n_group  = np.max(num_groups[bidx_st:bidx_ed])

                # proc groups
                mems = tuple()
                for gidx in range(n_group):
                    group_x = batch_x[:, gidx, :]
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]
                    
                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # (seq_len, bsz)
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()
                    
                    ret = model(group_x, group_y, group_mask, *mems)
                    loss, mems = ret[0], ret[1:]              
                    train_loss.append(loss.item()) 
                    loss.backward()

                    sys.stdout.write('epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}, group: {:2d}/{:2d} | Loss: {:6f}\r'.format(
                        epoch,
                        trainConfig['num_epochs'],
                        bidx,
                        num_batches,
                        gidx,
                        n_group, 
                        loss.item()
                    ))
                    sys.stdout.flush()

                optimizer.step()

            #val_loss = self.validate(val_data, batch_size, model, trainConfig["seed"], trainConfig['max_eval_steps'])
            curr_train_loss = sum(train_loss) / len(train_loss)
            saver_agent.add_summary('epoch loss', curr_train_loss)

            #epoch_val_loss.append(val_loss)
            epoch_train_loss.append(curr_train_loss)
            # epoch_info = 'Train Loss: {:.5f} , Val Loss: {:.5f}, T: {:.3f}'.format(curr_train_loss, val_loss, time.time()-st_time)
            epoch_info = 'Epoch: {}, Train Loss: {:.5f} ,  T: {:.3f}'.format(epoch+1, curr_train_loss, time.time()-st_time)
            print(epoch_info)

            # self.train_loss_record(epoch, curr_train_loss, checkpoint_dir, val_loss)
            self.train_loss_record(epoch, curr_train_loss, checkpoint_dir)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_setting': self.modelConfig,
                    'train_setting': trainConfig,
                    'state_dict': model.state_dict(),
                    'best_loss': curr_train_loss,
                    'optimizer' : optimizer.state_dict(),
                                }, 
                    checkpoint_dir, 
                    save_freq)
            
            wandb.log({'train/loss': curr_train_loss, 'epoch': epoch+1, 'time/epoch': time.time()-st_time}) 

            if curr_train_loss < 0.01:
                print('Experiment [{}] finished at loss < 0.01.'.format(checkpoint_dir))
                wandb.finish()
                break
            
        wandb.finish()

    def inference(self, model_path, token_lim, strategies, params, bpm, output_path, extra_output_paths=None):
        _, model = self.get_model(model_path)
        model.eval()
        
        # initial start
        words = [[]]

        # add beat
        words[-1].append(self.event2word['Bar_None'])
        
        # initialize mem
        mems = tuple()
        song_init_time = time.time()
        # generate
        initial_flag = True
        generate_n_bar = 0
        batch_size = 1
        cnt_bar = 0
        n_tokens = len(words[0])
        while len(words[0]) < token_lim:
            # prepare input
            if initial_flag:
                temp_x = np.zeros((len(words[0]), batch_size))

                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[z][b] = t
                
                initial_flag = False
            else:
                temp_x = np.zeros((1, batch_size))
                
                for b in range(batch_size):
                    temp_x[0][b] = words[b][-1] ####?####

            temp_x = torch.from_numpy(temp_x).long().to(self.device)     
            st_time = time.time()
            
            _logits, mems = model.generate(temp_x, *mems)
            logits = _logits.cpu().squeeze().detach().numpy()

            # temperature or not
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
                
            else:
                probs = self.temperature(logits=logits, temperature=1.)
            # sampling
            word = self.nucleus(probs=probs, p=params['p']) 
            print(word)  
            words[0].append(word)
            
            print(len(words[0]), self.word2event[word])
            # record n_bar
            if word == self.event2word['Bar_None']:
                generate_n_bar += 1
                
            if word == 0:
                cnt_bar += 1
                
            if cnt_bar >= 32:
                break
            

        wrtie_midi(words[0], output_path, self.word2event, extra_paths=extra_output_paths)

        song_total_time = time.time() - song_init_time
        print('Total words generated: ', len(words[0]))
        return song_total_time, len(words[0])

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word
