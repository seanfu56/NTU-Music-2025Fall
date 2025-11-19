import numpy as np
from glob import glob
import random, itertools
import pickle
import utils
import pandas as pd
import os
import scipy.stats
import tqdm
from tqdm import tqdm
import argparse
from miditoolkit import MidiFile
import json

from mueller import (
#   get_event_seq, 
  get_bars_crop, 
  get_pitch_histogram, 
  compute_histogram_entropy, 
  get_onset_xor_distance,
  get_chord_sequence,
  read_fitness_mat
)


class SequenceAdapter:
    """Convert different vocabularies into a common event stream."""

    BAR_CANONICAL = 10000
    POS_BASE = 11000

    def __init__(self, beat_labels=None):
        if not beat_labels:
            beat_labels = [f'Beat_{i}' for i in range(16)]

        beat_labels = sorted(set(beat_labels), key=self._beat_sort_key)
        self.pos_id_map = {
            label: self.POS_BASE + idx for idx, label in enumerate(beat_labels)
        }

        self.bar_ev_id = self.BAR_CANONICAL
        self.pos_evs = list(self.pos_id_map.values())
        self.pitch_evs = list(range(0, 128))

    @staticmethod
    def _beat_sort_key(label):
        try:
            return int(label.split('_')[1])
        except Exception:
            return 0

    def _append_metrical_label(self, label, seq):
        if not isinstance(label, str):
            return
        lower_label = label.lower()
        if lower_label.startswith('bar'):
            seq.append(self.bar_ev_id)
        elif label.startswith('Beat_'):
            pos_id = self.pos_id_map.get(label)
            if pos_id is not None:
                seq.append(pos_id)

    @staticmethod
    def _parse_pitch(label):
        if isinstance(label, str) and 'Note_Pitch_' in label:
            try:
                return int(label.split('_')[-1])
            except ValueError:
                return None
        return None

    def _append_pitch_label(self, label, seq):
        midi = self._parse_pitch(label)
        if midi is not None:
            seq.append(midi)

    def convert_sequence(self, seq):
        raise NotImplementedError


class RemiAdapter(SequenceAdapter):
    def __init__(self, event2word, word2event, beat_labels):
        super().__init__(beat_labels)
        self.id2event = word2event

    def convert_sequence(self, seq):
        events = []
        for ev in seq:
            label = self.id2event.get(ev)
            if label is None:
                continue
            if label.startswith('Note_Pitch_'):
                self._append_pitch_label(label, events)
            else:
                self._append_metrical_label(label, events)
        return events


class CPWordAdapter(SequenceAdapter):
    def __init__(self, event2word, word2event, beat_labels):
        super().__init__(beat_labels)
        class_keys = list(word2event.keys())
        self.key_to_idx = {key: idx for idx, key in enumerate(class_keys)}
        self.decoders = {key: word2event[key] for key in class_keys}

    def _decode(self, key, token):
        idx = self.key_to_idx.get(key)
        decoder = self.decoders.get(key)
        if idx is None or decoder is None:
            return None
        val = token[idx]
        return decoder.get(val)

    def convert_sequence(self, seq):
        events = []
        for token in seq:
            typ = self._decode('type', token)
            if typ == 'Metrical':
                label = self._decode('bar-beat', token)
                self._append_metrical_label(label, events)
            elif typ == 'Note':
                pitch = self._decode('pitch', token)
                self._append_pitch_label(pitch, events)
        return events


def build_adapter(event2word, word2event):
    try:
        first_val = next(iter(event2word.values()))
    except StopIteration:
        raise ValueError('dictionary is empty, cannot build adapter')

    if isinstance(first_val, dict):
        beat_labels = [
            key for key in event2word.get('bar-beat', {})
            if isinstance(key, str) and key.startswith('Beat_')
        ]
        return CPWordAdapter(event2word, word2event, beat_labels)

    beat_labels = [
        key for key in event2word.keys()
        if isinstance(key, str) and key.startswith('Beat_')
    ]
    return RemiAdapter(event2word, word2event, beat_labels)


#############################################################################
'''Dynamic event encodings, resolved after loading dictionary.'''
BAR_EV = None
POS_EVS = None
PITCH_EVS = None
adapter = None
#############################################################################

def parse_opt():
    parser = argparse.ArgumentParser()
    # training opts
    parser.add_argument('-d', '--dict_path', type=str,
                        help='the dictionary path', required=True)
    parser.add_argument('-o', '--output_file_path', type=str,
                        help='the output file path.', required=True)
    args = parser.parse_args()
    return args
  
opt = parse_opt()


event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
adapter = build_adapter(event2word, word2event)
BAR_EV = adapter.bar_ev_id
POS_EVS = adapter.pos_evs
PITCH_EVS = adapter.pitch_evs

print(event2word.keys())


def extract_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end

    items = tempo_items + note_items

    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    return events

def prepare_data(midi_path):
    # extract events
    
    with open(midi_path, 'r') as f:
        words = json.load(f)

    if adapter is None:
        raise RuntimeError('sequence adapter is not initialized')

    return adapter.convert_sequence(words)

def compute_piece_pitch_entropy(piece_ev_seq, window_size, bar_ev_id=BAR_EV, pitch_evs=PITCH_EVS, verbose=False):
  '''
  Computes the average pitch-class histogram entropy of a piece.
  (Metric ``H``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    window_size (int): length of segment (in bars) involved in the calc. of entropy at once.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when a crop contains no notes.

  Returns:
    float: the average n-bar pitch-class histogram entropy of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  n_bars = piece_ev_seq.count(bar_ev_id)
  if window_size > n_bars:
    print ('[Warning] window_size: {} too large for the piece, falling back to #(bars) of the piece.'.format(window_size))
    window_size = n_bars

  # compute entropy of all possible segments
  pitch_ents = []
  for st_bar in range(0, n_bars - window_size + 1):
    seg_ev_seq = get_bars_crop(piece_ev_seq, st_bar, st_bar + window_size - 1, bar_ev_id)

    pitch_hist = get_pitch_histogram(seg_ev_seq, pitch_evs=pitch_evs)
    if pitch_hist is None:
      if verbose:
        print ('[Info] No notes in this crop: {}~{} bars.'.format(st_bar, st_bar + window_size - 1))
      continue

    pitch_ents.append( compute_histogram_entropy(pitch_hist) )

  return np.mean(pitch_ents)

def compute_piece_groove_similarity(piece_ev_seq, bar_ev_id=BAR_EV, pos_evs=POS_EVS, pitch_evs=PITCH_EVS, max_pairs=1000):
  '''
  Computes the average grooving pattern similarity between all pairs of bars of a piece.
  (Metric ``GS``)

  Parameters:
    piece_ev_seq (list): a piece of music in event sequence representation.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    max_pairs (int): maximum #(pairs) considered, to save computation overhead.

  Returns:
    float: 0~1, the average grooving pattern similarity of the input piece.
  '''
  # remove redundant ``Bar`` marker
  if piece_ev_seq[-1] == bar_ev_id:
    piece_ev_seq = piece_ev_seq[:-1]

  # get every single bar & compute indices of bar pairs
  n_bars = piece_ev_seq.count(bar_ev_id)
  bar_seqs = []
  for b in range(n_bars):
    bar_seqs.append( get_bars_crop(piece_ev_seq, b, b, bar_ev_id) )
  pairs = list( itertools.combinations(range(n_bars), 2) )
  if len(pairs) > max_pairs:
    pairs = random.sample(pairs, max_pairs)

  # compute pairwise grooving similarities
  grv_sims = []
  for p in pairs:
    grv_sims.append(
      1. - get_onset_xor_distance(bar_seqs[p[0]], bar_seqs[p[1]], bar_ev_id, pos_evs, pitch_evs=pitch_evs)
    )

  return np.mean(grv_sims)


if __name__ == "__main__":
  # codes below are for testing
  test_pieces = sorted(glob(os.path.join(opt.output_file_path, '*.json')))

  # print (test_pieces)

  result_dict = {
      'piece_name': [],
      'H1': [],
      'H4': [],
      'GS': []
  }

  for p in tqdm(test_pieces):
      result_dict['piece_name'].append(p.replace('\\', '/').split('/')[-1])
      seq = prepare_data(p)

      h1 = compute_piece_pitch_entropy(seq, 1)
      result_dict['H1'].append(h1)
      h4 = compute_piece_pitch_entropy(seq, 4)
      result_dict['H4'].append(h4)
      gs = compute_piece_groove_similarity(seq)
      result_dict['GS'].append(gs)

  if len(result_dict):
      df = pd.DataFrame.from_dict(result_dict)
      # calculate average
      avg_dict = {
        'piece_name': 'Average',
        'H1': np.mean([v for v in result_dict['H1'] if v is not None]),
        'H4': np.mean([v for v in result_dict['H4'] if v is not None]),
        'GS': np.mean([v for v in result_dict['GS'] if v is not None]),
      }
      # add avg_dict to df
      df = df._append(avg_dict, ignore_index=True)
      df.to_csv(f'{opt.output_file_path}/eval.csv', index=False, encoding='utf-8')
