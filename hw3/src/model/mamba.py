import sys
import os
import time
import math
import json
import logging
import collections
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import matplotlib.pyplot as plt
import wandb

from sklearn.model_selection import train_test_split

# ================================ #
# 常數設定
# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}

# ================================ #
# Saver：負責 log / 存 model
# ================================ #

class Saver(object):
    def __init__(self, exp_dir, mode='w'):
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
            filemode=mode
        )
        self.logger = logging.getLogger('training monitor')

    def add_summary_msg(self, msg):
        self.logger.debug(msg)

    def add_summary(self, key, val, step=None, cur_time=None):
        if cur_time is None:
            cur_time = time.time() - self.init_time
        if step is None:
            step = self.global_step

        if isinstance(val, float):
            msg_str = '{:10s} | {:.10f} | {:10d} | {}'.format(
                key, val, step, cur_time
            )
        else:
            msg_str = '{:10s} | {} | {:10d} | {}'.format(
                key, val, step, cur_time
            )

        self.logger.debug(msg_str)

    def save_model(self, model, optimizer=None, outdir=None, name='model'):
        if outdir is None:
            outdir = self.exp_dir
        print(' [*] saving model to {}, name: {}'.format(outdir, name))
        torch.save(model.state_dict(), os.path.join(outdir, name + '_params.pt'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(outdir, name + '_opt.pt'))

    def load_model(self, path_exp, device='cpu', name='model_params.pt'):
        path_pt = os.path.join(path_exp, name)
        print(' [*] restoring model from', path_pt)
        state_dict = torch.load(path_pt, map_location=torch.device(device))
        return state_dict

    def global_step_increment(self):
        self.global_step += 1


def make_loss_report(path_log, path_figure='loss.png', dpi=100):
    """
    讀 log.txt，畫出 train / valid loss 的曲線。
    path_log: log 檔案路徑 (通常是 exp_dir/log.txt)
    """
    monitor_vals = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    if 'train loss' not in monitor_vals or 'valid loss' not in monitor_vals:
        print("log 檔裡沒有 'train loss' / 'valid loss'，無法畫圖")
        return

    step_train = [item[1] for item in monitor_vals['train loss']]
    vals_train = [item[0] for item in monitor_vals['train loss']]

    step_valid = [item[1] for item in monitor_vals['valid loss']]
    vals_valid = [item[0] for item in monitor_vals['valid loss']]

    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure)


# ================================ #
# MIDI 寫檔
# ================================ #

def wrtie_midi(words, path_midi, word2event, extra_paths=None):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile()
    cur_pos = 0

    for i in range(len(events) - 3):
        cur_event = events[i]
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
               'Note_Velocity' in events[i + 1] and \
               'Note_Duration' in events[i + 2]:

                pitch = int(events[i].split('_')[-1])
                duration = int(events[i + 2].split('_')[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration
                velocity = int(events[i + 1].split('_')[-1])
                notes_all.append(
                    Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))

    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)

    # 也把 token 序列存一份
    with open(path_midi.replace('.mid', '.json'), 'w') as f:
        # output words
        json.dump([int(word) for word in words], f)

# ================================ #
# 工具：計算參數量
# ================================ #

def network_paras(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


# ================================ #
# Mamba-based LM
# ================================ #

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "請先安裝 mamba-ssm 套件：\n"
        "  pip install mamba-ssm\n"
    )


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T) or (T, B)
        if x.dim() == 2 and x.size(0) > x.size(1):  # (T,B) -> (B,T)
            x = x.transpose(0, 1)
        return self.lut(x) * math.sqrt(self.d_model)


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


class MambaLM(nn.Module):
    """
    用 Mamba 做 sequence modeling 的語言模型。
    介面盡量模仿原本的 MemTransformerLM：
      - forward(data, target, mask, *mems) -> [loss]
      - generate(data, *mems) -> (logits_last_step, new_mems)
    只是這裡不再真的使用 mems（Mamba 本身就擅長 long sequence）
    """
    def __init__(self, modelConfig, is_training=True):
        super().__init__()

        self.n_token = modelConfig['n_token']
        self.n_layer = modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.d_embed = self.d_model if modelConfig.get('d_embed') is None else modelConfig['d_embed']

        self.dropout = modelConfig.get('dropout', 0.1)
        self.d_inner = modelConfig.get('d_inner', 4 * self.d_model)

        # Mamba 相關的 hyperparameter（若沒有就用預設）
        self.d_state = modelConfig.get('d_state', 16)
        self.d_conv = modelConfig.get('d_conv', 4)
        self.expand = modelConfig.get('expand', 2)

        self.is_training = is_training

        # embedding
        self.word_emb = Embeddings(self.n_token, self.d_model)
        self.drop = nn.Dropout(self.dropout)

        # 疊多層 Mamba block
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                d_inner=self.d_inner,
                dropout=self.dropout
            )
            for _ in range(self.n_layer)
        ])

        self.norm_out = nn.LayerNorm(self.d_model)

        # output projection
        self.linear_proj = nn.Linear(self.d_model, self.n_token)

        # loss
        self.loss_func = nn.CrossEntropyLoss(reduction='none')


    def _forward_hidden(self, data):
        """
        data: (T, B) or (B, T) with token indices
        return: hidden: (B, T, d_model)
        """
        if data.dim() != 2:
            raise ValueError("data 應該是 2D (T,B) or (B,T)")

        # 統一成 (B, T)
        if data.size(0) > data.size(1):  # (T, B)
            data = data.transpose(0, 1)   # -> (B, T)

        x = self.word_emb(data)          # (B, T, d_model)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        x = self.drop(x)
        return x

    def generate(self, data, *mems):
        """
        data: (T, B) 或 (B, T)，只用來當 prefix，輸出最後一個時間步的 logits。
        回傳:
          - predict: shape (1, B, n_token)（模仿原本 TransformerXL 的介面）
          - new_mems: 空 tuple（這裡沒有真的使用 mems）
        """
        hidden = self._forward_hidden(data)     # (B, T, d_model)
        last_hidden = hidden[:, -1:, :]         # (B, 1, d_model)
        logits = self.linear_proj(last_hidden)  # (B, 1, n_token)
        logits = logits.permute(1, 0, 2)        # (1, B, n_token) 方便跟原版一致
        new_mems = tuple()
        return logits, new_mems

    def compute_loss(self, predict, target, loss_mask=None):
        """
        predict: (B, C, T)
        target:  (B, T)
        loss_mask: (B, T)
        """
        B, C, T = predict.size()

        # (B,C,T) -> (B,T,C) -> (B*T, C)
        predict_flat = predict.permute(0, 2, 1).contiguous().view(-1, C)

        target = target.contiguous()
        target_flat = target.view(-1).long()  # (B*T,)

        loss_all = self.loss_func(predict_flat, target_flat)  # (B*T,)

        if loss_mask is not None:
            mask_flat = loss_mask.contiguous().view(-1)
            loss_all = loss_all * mask_flat
            loss = torch.sum(loss_all) / (torch.sum(mask_flat) + 1e-8)
        else:
            loss = torch.mean(loss_all)

        return loss

    def forward(self, data, target, mask, *mems):
        """
        data:   (T, B)
        target: (T, B)
        mask:   (B, T)
        回傳 [loss]
        """
        hidden = self._forward_hidden(data)      # (B, T, d_model)

        logits = self.linear_proj(hidden)        # (B, T, n_token)
        logits = logits.permute(0, 2, 1)         # (B, n_token, T)

        if target.size(0) > target.size(1):      # (T,B) -> (B,T)
            target = target.transpose(0, 1)
        target = target.contiguous()

        loss = self.compute_loss(logits, target, mask)

        return [loss]


# ================================ #
# 包裝：原本叫 TransformerXL，現在裡面改成 MambaLM
# ================================ #

class TransformerXL(object):
    """
    名字沿用 TransformerXL，但裡面的 model 已經換成 MambaLM。
    用法：
        txl = TransformerXL(modelConfig, device, event2word, word2event)
        txl.train(train_data, trainConfig, device, resume)
        txl.inference(...)
    """
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings（有些是從舊版沿用，Mamba 不一定用得到）
        self.n_layer = modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len = modelConfig.get('seq_len', 512)

        self.mem_len = modelConfig.get('mem_len', 0)
        self.tgt_len = modelConfig.get('tgt_len', self.seq_len)
        self.ext_len = modelConfig.get('ext_len', 0)
        self.eval_tgt_len = modelConfig.get('eval_tgt_len', self.tgt_len)

        self.init = modelConfig.get('init', 'normal')
        self.init_range = modelConfig.get('init_range', 0.1)
        self.init_std = modelConfig.get('init_std', 0.02)
        self.proj_init_std = modelConfig.get('proj_init_std', 0.02)

        self.is_training = is_training
        self.device = device

    # -------- 初始化權重 --------
    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(self, m):
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

    # -------- 建 model / 載 checkpoint --------
    def get_model(self, pretrain_model=None):
        model = MambaLM(self.modelConfig, is_training=self.is_training)

        st_epoch = 0
        if pretrain_model and pretrain_model != 'None':
            checkpoint = torch.load(pretrain_model, map_location=self.device)
            print('Pretrained model config:')
            print('epoch: ', checkpoint['epoch'])
            print('best_loss: ', checkpoint['best_loss'])
            print(json.dumps(checkpoint['model_setting'], indent=1, sort_keys=True))
            print(json.dumps(checkpoint['train_setting'], indent=1, sort_keys=True))

            try:
                model.load_state_dict(checkpoint['state_dict'])
                print('{} loaded.'.format(pretrain_model))
            except Exception as e:
                print('Loaded weights have different shapes with the model.')
                print('Error:', e)
                sys.exit(1)
            st_epoch = checkpoint['epoch'] + 1
        else:
            self.init_weight(model.word_emb.lut.weight)
            self.init_weight(model.linear_proj.weight)
            if model.linear_proj.bias is not None:
                nn.init.zeros_(model.linear_proj.bias)
            pass

        return st_epoch, model.to(self.device)

    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root, 'ep_{}.pth.tar'.format(state['epoch'])))

    def train_loss_record(self, epoch, train_loss, checkpoint_dir, val_loss=None):
        if val_loss is not None:
            df = pd.DataFrame({
                'epoch': [epoch + 1],
                'train_loss': ['%.3f' % train_loss],
                'val_loss': ['%.3f' % val_loss]
            })
        else:
            df = pd.DataFrame({
                'epoch': [epoch + 1],
                'train_loss': ['%.3f' % train_loss]
            })

        csv_file = os.path.join(checkpoint_dir, 'loss.csv')

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)

    # ================================ #
    # 訓練
    # ================================ #
    def train(self, train_data, trainConfig, device, resume):
        checkpoint_dir = trainConfig['experiment_Dir']
        batch_size = trainConfig['batch_size']
        torch.manual_seed(trainConfig["seed"])

        saver_agent = Saver(checkpoint_dir)

        wandb_config = {
            'model_config': self.modelConfig,
            'train_config': trainConfig
        }

        wandb.init(
            project="music-hw3-cp",
            name="mamba_remi",
            config=wandb_config
        )

        

        # 準備 model
        if resume != 'None':
            st_epoch, model = self.get_model(resume)
            print('Continue to train from {} epoch'.format(st_epoch))
        else:
            st_epoch, model = self.get_model()

        optimizer = optim.Adam(model.parameters(), lr=trainConfig['lr'])
        epoch_train_loss = []
        save_freq = trainConfig['save_freq']

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainConfig['num_epochs'], eta_min=0)
        n_parameters = network_paras(model)
        print('n_parameters: {:,}'.format(n_parameters))
        saver_agent.add_summary_msg(
            ' > params amount: {:,d}'.format(n_parameters))

        # unpack data
        train_x = train_data['x']
        train_y = train_data['y']
        mask = train_data['mask']
        num_groups = train_data['num_groups']

        num_batches = len(train_x) // batch_size

        print('>>> Start training')
        for epoch in range(st_epoch, trainConfig['num_epochs']):
            saver_agent.global_step_increment()

            train_loss = []
            st_time = time.time()
            model.train()

            for bidx in range(num_batches):
                model.zero_grad()

                bidx_st = batch_size * bidx
                bidx_ed = batch_size * (bidx + 1)

                batch_x = train_x[bidx_st:bidx_ed]
                batch_y = train_y[bidx_st:bidx_ed]
                batch_mask = mask[bidx_st:bidx_ed]
                n_group = int(np.max(num_groups[bidx_st:bidx_ed]))

                # Mamba 不使用 mems，但我們可以照樣寫 interface
                mems = tuple()

                for gidx in range(n_group):
                    group_x = batch_x[:, gidx, :]   # (B, T)
                    group_y = batch_y[:, gidx, :]
                    group_mask = batch_mask[:, gidx, :]  # (B, T)

                    group_x = torch.from_numpy(group_x).permute(1, 0).contiguous().to(self.device).long()  # (T,B)
                    group_y = torch.from_numpy(group_y).permute(1, 0).contiguous().to(self.device).long()  # (T,B)
                    group_mask = torch.from_numpy(group_mask).to(self.device).float()                       # (B,T)

                    ret = model(group_x, group_y, group_mask, *mems)
                    loss = ret[0]
                    train_loss.append(loss.item())
                    loss.backward()

                    sys.stdout.write(
                        'epoch:{:3d}/{:3d}, batch: {:4d}/{:4d}, group: {:2d}/{:2d} | Loss: {:6f}\r'.format(
                            epoch,
                            trainConfig['num_epochs'],
                            bidx,
                            num_batches,
                            gidx,
                            n_group,
                            loss.item()
                        )
                    )
                    sys.stdout.flush()

                optimizer.step()

            curr_train_loss = sum(train_loss) / len(train_loss)
            saver_agent.add_summary('epoch loss', curr_train_loss)

            epoch_train_loss.append(curr_train_loss)
            epoch_info = 'Epoch: {}, Train Loss: {:.5f} ,  T: {:.3f}'.format(
                epoch + 1, curr_train_loss, time.time() - st_time)
            print('\n' + epoch_info)

            self.train_loss_record(epoch, curr_train_loss, checkpoint_dir)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'model_setting': self.modelConfig,
                'train_setting': trainConfig,
                'state_dict': model.state_dict(),
                'best_loss': curr_train_loss,
                'optimizer': optimizer.state_dict(),
            },
                checkpoint_dir,
                save_freq
            )

            scheduler.step()

            wandb.log({'train/loss': curr_train_loss,
                       'epoch': epoch + 1,
                       'time/epoch': time.time() - st_time})

            if curr_train_loss < 0.01:
                print('Experiment [{}] finished at loss < 0.01.'.format(checkpoint_dir))
                wandb.finish()
                break

        wandb.finish()

    # ================================ #
    # 推論 / 產生 MIDI
    # ================================ #
    def inference(self, model_path, token_lim, strategies, params, bpm,
                  output_path, extra_output_paths=None):
        print(model_path)
        _, model = self.get_model(model_path)
        model.eval()

        # initial start
        words = [[]]

        # add first bar
        words[-1].append(self.event2word['Bar_None'])

        mems = tuple()
        use_full_context = True
        song_init_time = time.time()
        batch_size = 1
        cnt_bar = 0

        while len(words[0]) < token_lim:
            # 準備 input
            if use_full_context:
                seq_len = len(words[0])
                temp_x = np.zeros((seq_len, batch_size))
                for b in range(batch_size):
                    temp_x[:, b] = words[b]
            else:
                if len(words[0]) == 1:
                    temp_x = np.zeros((len(words[0]), batch_size))
                    for b in range(batch_size):
                        for z, t in enumerate(words[b]):
                            temp_x[z][b] = t
                else:
                    temp_x = np.zeros((1, batch_size))
                    for b in range(batch_size):
                        temp_x[0][b] = words[b][-1]

            temp_x = torch.from_numpy(temp_x).long().to(self.device)  # (T,B)

            _logits, mems = model.generate(temp_x, *mems)
            logits = _logits.cpu().squeeze(0).detach().numpy()   # (B, vocab)
            logits = logits[0]                                   # (vocab,)

            # temperature
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
            else:
                probs = self.temperature(logits=logits, temperature=1.0)

            # nucleus sampling
            word = self.nucleus(probs=probs, p=params['p'])
            words[0].append(word)
            
            if word == 0:
                cnt_bar += 1
                
            if cnt_bar >= 32:
                break   
            
            print(len(words[0]), self.word2event[word])

        wrtie_midi(words[0], output_path, self.word2event, extra_paths=extra_output_paths)

        song_total_time = time.time() - song_init_time
        print('Total words generated: ', len(words[0]))
        return song_total_time, len(words[0])

    # ================================ #
    # sampling strategies
    # ================================ #
    def temperature(self, logits, temperature):
        logits = logits / temperature
        logits = logits - np.max(logits)  # 防止 overflow
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return probs

    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs = np.array(candi_probs) / np.sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    def nucleus(self, probs, p):
        probs = np.array(probs, dtype=np.float64)
        probs /= probs.sum()

        sorted_index = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_index]
        cusum_sorted_probs = np.cumsum(sorted_probs)

        after_threshold = cusum_sorted_probs > p
        if np.sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3]

        candi_probs = probs[candi_index]
        candi_probs /= candi_probs.sum()
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word
