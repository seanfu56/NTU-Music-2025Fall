import librosa

import numpy as np
import librosa 
import math
import tqdm

'''
Source --
  * https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_AudioThumbnailing.html

Authored by: 
  Meinard Mueller, Angel Villar-Corrales
Arranged by:
  Wen-Yi Hsiao, Shih-Lun Wu
'''

# ------------------------------------------------------------ #
# Fitness Scape Plot Computation
# ------------------------------------------------------------ #
def normalization_properties_SSM(S):
    """Normalizes self-similartiy matrix to fulfill S(n,n)=1
    Yields a warning if max(S)<=1 is not fulfilled
   
    Notebook: C4/C4S3_AudioThumbnailing.ipynb 
    """    
    N = S.shape[0]
    for n in range(N): 
        S[n,n] = 1
        max_S = np.max(S)
    if max_S>1:
        print('Normalization condition for SSM not fulfill (max > 1)')
    return S


def compute_accumulated_score_matrix(S_seg):
    """Compute the accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S_seg: submatrix of an enhanced and normalized SSM S 
                 Note: S must satisfy S(n,m) <= 1 and S(n,n) = 1
        
    Returns:
        D: Accumulated score matrix 
        score: Score of optimal path family 
    """
    inf = math.inf  
    N =  S_seg.shape[0]
    M =  S_seg.shape[1]+1
    
    # Iinitializing score matrix
    D = -inf*np.ones((N,M), dtype=np.float64)
    D[0,0] = 0.
    D[0,1] = D[0,0]+S_seg[0,0]

    # Dynamic programming
    for n in range(1, N):
        D[n,0] = max( D[n-1,0], D[n-1,-1] )    
        D[n,1] = D[n,0] + S_seg[n, 0]
        for m in range(2, M):
            D[n, m] = S_seg[n, m-1] + max( D[n-1, m-1], D[n-1, m-2], D[n-2, m-1] )
            
    # Score of optimal path family
    score = np.maximum( D[N-1,0], D[N-1,M-1] )
    
    return D, score

def compute_optimal_path_family(D):
    """Compute an optimal path family given an accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        D: Accumulated score matrix

    Returns
        P: Optimal path family consisting of list of paths 
           (each path being a list of index pairs)
    """
    # Initialization
    inf = math.inf
    N = int(D.shape[0])
    M = int(D.shape[1])
    
    path_family = []
    path = []
    
    n = N - 1
    if( D[n,M-1]<D[n,0] ):
        m = 0
    else:
        m = M-1
        path_point = (N-1, M-2)
        path.append(path_point)
    
    # Backtracking 
    while n > 0 or m > 0:

        # obtaining the set of possible predecesors given our current position
        if(n<=2 and m<=2):
            predecessors = [(n-1,m-1)]
        elif(n<=2 and m>2):
            predecessors = [(n-1,m-1),(n-1,m-2)]
        elif(n>2 and m<=2):
            predecessors = [(n-1,m-1),(n-2,m-1)]
        else:
            predecessors = [(n-1,m-1),(n-2,m-1),(n-1,m-2)]
        
        # case for the first row. Only horizontal movements allowed
        if n == 0:
            cell = (0, m-1)
        # case for the elevator column: we can keep going down the column or jumping to the end of the next row
        elif m == 0:
            if( D[n-1, M-1] > D[n-1, 0] ):
                cell = (n-1, M-1)
                path_point = (n-1, M-2)
                if(len(path)>0):
                    path.reverse()
                    path_family.append(path)
                path = [path_point]
            else:
                cell = (n-1, 0)
        # case for m=1, only horizontal steps to the elevator column are allowed
        elif m == 1:
            cell=(n,0)          
        # regular case
        else:
        
            #obtaining the best of the possible predecesors
            max_val = -inf
            for i in range(len(predecessors)):
                if( max_val<D[predecessors[i][0],predecessors[i][1]] ):
                    max_val = D[predecessors[i][0],predecessors[i][1]]
                    cell = predecessors[i]  
                    
            #saving the point in the current path
            path_point = (cell[0],cell[1]-1)
            path.append(path_point)        
            
        (n, m) = cell
    
    # adding last path to the path family
    path.reverse()
    path_family.append(path)
    path_family.reverse()
    
    return path_family

def compute_induced_segment_family_coverage(path_family):
    """Compute induced segment family and coverage from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family

    Returns
        segment_family: Induced segment family
        coverage: Coverage of path family
    """
    num_path = len(path_family)
    coverage = 0
    if num_path>0:
        segment_family = np.zeros((num_path, 2), dtype=int)
        for n in range(num_path):
            segment_family[n,0] = path_family[n][0][0]
            segment_family[n,1] = path_family[n][-1][0]
            coverage = coverage + segment_family[n,1] - segment_family[n,0] + 1
    else:
        segment_family = np.empty
        
    return segment_family, coverage

def compute_fitness(path_family, score, N):
    """Compute fitness measure and other metrics from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family 
        score: Score of path family 
        N: Length of feature sequence

    Returns
        fitness: Fitness
        score: Score
        score_n: Normalized score
        coverage: Coverage
        coverage_n: Normalized coverage
        path_family_length: Length of path family (total number of cells)
    """
    eps = 1e-16
    num_path = len(path_family)
    M = path_family[0][-1][1] + 1
    
    # Normalized score
    path_family_length = 0 
    for n in range(num_path):
        path_family_length = path_family_length + len(path_family[n])
    score_n = (score - M) / (path_family_length + eps)

    # Normalized coverage
    segment_family, coverage = compute_induced_segment_family_coverage(path_family)
    coverage_n = (coverage - M) / (N + eps)
    
    # Fitness measure
    fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)
    
    return fitness, score, score_n, coverage, coverage_n, path_family_length


def compute_fitness_scape_plot(S):
    """Compute scape plot for fitness and other measures

    Notebook: /C4/C4S3_ScapePlot.ipynb

    Args:
        S: Self-similarity matrix 

    Returns:
        SP_all: Vector containing five different scape plots for five measures
            (fitness, score, normalized score, coverage, normlized coverage)
            (encoded as start-duration matrix)
    """
    N = S.shape[0]
    SP_fitness = np.zeros((N,N))    
    SP_score = np.zeros((N,N))
    SP_score_n = np.zeros((N,N))
    SP_coverage = np.zeros((N,N))
    SP_coverage_n = np.zeros((N,N))
    
    for length_minus_one in tqdm.tqdm(range(N)):
        for start in range(N-length_minus_one):
            S_seg = S[:,start:start+length_minus_one+1]         
            D, score = compute_accumulated_score_matrix(S_seg)
            path_family = compute_optimal_path_family(D)
            fitness, score, score_n, coverage, coverage_n, path_family_length = \
                compute_fitness(path_family, score, N)
            SP_fitness[length_minus_one,start]= fitness
            SP_score[length_minus_one,start]= score
            SP_score_n[length_minus_one,start]= score_n
            SP_coverage[length_minus_one,start]= coverage
            SP_coverage_n[length_minus_one,start]= coverage_n
    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all

import numpy as np
import multiprocessing as mp
import tqdm

# ====== 給 child process 用的全域變數 ======
_S_GLOBAL = None
_N_GLOBAL = None

def _init_worker(S, N):
    """在每個 child process 裡，把 S / N 存成全域，避免每個 task 都重傳一次大矩陣"""
    global _S_GLOBAL, _N_GLOBAL
    _S_GLOBAL = S
    _N_GLOBAL = N

def _compute_one_cell(args):
    """計算單一 (length_minus_one, start) 對應的 fitness / score / coverage 等"""
    length_minus_one, start = args
    S = _S_GLOBAL
    N = _N_GLOBAL

    # 取出對應 segment
    S_seg = S[:, start:start + length_minus_one + 1]

    D, score = compute_accumulated_score_matrix(S_seg)
    path_family = compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = \
        compute_fitness(path_family, score, N)

    return (length_minus_one, start,
            fitness, score, score_n, coverage, coverage_n)


def compute_fitness_scape_plot_parallel(S, num_workers=None, use_tqdm=True):
    """平行版的 fitness scape plot 計算

    Args:
        S: self-similarity matrix，shape = (N, N)
        num_workers: process 數量；None = 預設用 mp.cpu_count()
        use_tqdm: 是否顯示 progress bar

    Returns:
        SP_all: [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
                各自 shape = (N, N)
    """
    N = S.shape[0]

    SP_fitness   = np.zeros((N, N), dtype=np.float32)
    SP_score     = np.zeros((N, N), dtype=np.float32)
    SP_score_n   = np.zeros((N, N), dtype=np.float32)
    SP_coverage  = np.zeros((N, N), dtype=np.float32)
    SP_coverage_n= np.zeros((N, N), dtype=np.float32)

    # 建立所有需要計算的 (length_minus_one, start) 組合
    jobs = []
    for length_minus_one in range(N):
        for start in range(N - length_minus_one):
            jobs.append((length_minus_one, start))

    if num_workers is None:
        num_workers = mp.cpu_count()

    # 為了不要每個 task 都 pickle 一份 S，我們用 initializer 把 S 設成全域
    with mp.Pool(processes=num_workers,
                 initializer=_init_worker,
                 initargs=(S, N)) as pool:

        iterator = pool.imap_unordered(_compute_one_cell, jobs, chunksize=32)

        if use_tqdm:
            iterator = tqdm.tqdm(iterator, total=len(jobs), desc="ScapePlot")

        for (length_minus_one, start,
             fitness, score, score_n, coverage, coverage_n) in iterator:

            SP_fitness[length_minus_one, start]    = fitness
            SP_score[length_minus_one, start]      = score
            SP_score_n[length_minus_one, start]    = score_n
            SP_coverage[length_minus_one, start]   = coverage
            SP_coverage_n[length_minus_one, start] = coverage_n

    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all

import librosa

import numpy as np
import librosa 
from scipy import signal

'''
Source --
  * https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3.html
  * https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4.html

Authored by:
  Meinard Mueller, David Kopyto, Vlora Arifi-Mueller
Arranged by:
  Wen-Yi Hsiao, Shih-Lun Wu
'''

# ------------------------------------------------------------ #
# Audio Feature Processing (e.g., pitch class profile)
# ------------------------------------------------------------ #
def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X: Feature sequence
        norm: The norm to be applied. '1', '2', 'max' or 'z'
        threshold: An threshold below which the vector `v` used instead of normalization
        v: Used instead of normalization below `threshold`. If None, uses unit vector for given norm

    Returns:
        X_norm: Normalized feature sequence
    """
    K, N = X.shape
    X_norm = np.zeros((K, N))
    if norm == '1':
        if v is None:
            v = np.ones(K) / K 
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == '2':
        if v is None:
            v = np.ones(K) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == 'max':
        if v is None:
            v = np.ones(K)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == 'z':
        if v is None:
            v = np.zeros(K)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v
    return X_norm


def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    Notebook: C3/C3S1_FeatureSmoothing.ipynb

    Args:
        X: Feature sequence
        Fs: Frame rate of `X`
        filt_len: Length of smoothing filter
        down_sampling: Downsampling factor
        w_type: Window type of smoothing filter

    Returns:
        X_smooth: Smoothed and downsampled feature sequence
        Fs_feature: Frame rate of `X_smooth`
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature


# ------------------------------------------------------------ #
# Self-similarity Matrix Computation & Enhancement
# ------------------------------------------------------------ #
def compute_SM_dot(X,Y):
    """Computes similarty matrix from feature sequences using dot (inner) product
    Notebook: C4/C4S2_SSM.ipynb
    """    
    S = np.dot(np.transpose(Y),X)    
    return S


def filter_diag_mult_SM(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):   
    """Path smoothing of similarity matrix by filtering in forward or backward direction 
    along various directions around main diagonal
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Self-similarity matrix (SSM)
        L: Length of filter 
        tempo_rel_set: Set of relative tempo values
        direction: Direction of smoothing (0: forward; 1: backward)

    Returns:
        S_L_final: Smoothed SM   
    """        
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((M,N))
    
    for s in range(0, num):
        M_ceil = int(np.ceil(N/tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1,M_ceil+1),M_ceil),N)
        np.around(resample, 0, resample)
        resample = resample -1        
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:,index_resample]
            
        S_L = np.zeros((M,M_ceil))
        S_extend_L = np.zeros((M + L, M_ceil + L))
        
        # Forward direction
        if direction==0:
            S_extend_L[0:M,0:M_ceil] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[pos:(M + pos), pos:(M_ceil + pos)]    
                
        # Backward direction        
        if direction==1:
            S_extend_L[L:(M+L),L:(M_ceil+L)] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[(L-pos):(M + L - pos), (L-pos):(M_ceil + L - pos)]      
    
        S_L = S_L/L    
        resample = np.multiply(np.divide(np.arange(1,N+1),N),M_ceil)
        np.around(resample, 0, resample)
        resample = resample-1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)    
        
        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)
    return S_L_final


def compute_tempo_rel_set(tempo_rel_min, tempo_rel_max, num):
    """Compute logarithmically spaced relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        tempo_rel_min: Minimum relative tempo
        tempo_rel_max: Maximum relative tempo 
        num: Number of relative tempo values (inlcuding the min and max)

    Returns:
        tempo_rel_set: Set of relative tempo values
    """
    tempo_rel_set = np.exp(np.linspace(np.log(tempo_rel_min), np.log(tempo_rel_max), num))
    return tempo_rel_set



def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X: Feature respresentation
        shift: Number of bins to be shifted

    Returns:
        X_cyc: Cyclically shifted feature matrix
    """
    #Note: X_cyc = np.roll(X, shift=shift, axis=0) does to work for jit
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K,N))
    X_cyc[shift:K, :] = X[0:K-shift, :] 
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc



def compute_SM_TI(X, Y, L=1, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions 

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X, Y: Input feature sequences 
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        shift_set: Set of shift indices
        direction: Direction of smoothing (0: forward; 1: backward; 2: both directions)

    Returns:
        S_TI: Transposition-invariant SM
        I_TI: Transposition index matrix
    """
    for shift in shift_set:
        X_cyc = shift_cyc_matrix(X, shift)
        S_cyc = compute_SM_dot(X,X_cyc)

        if direction==0:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=0)
        if direction==1:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=1)
        if direction==2:
            S_forward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift ==  shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0],S_cyc.shape[1])) * shift
        else:
            #jit does not like the following lines
            #I_greater = np.greater(S_cyc, S_TI)
            #I_greater = (S_cyc>S_TI)
            I_TI[S_cyc>S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)
    return S_TI, I_TI


def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0, binarize=False):
    """Threshold matrix in a relative fashion 

    Notebook: C4/C4/C4S2_SSM-Thresholding.ipynb

    Args:
        S: Input matrix
        thresh: Threshold (meaning depends on strategy)
        strategy: Thresholding strategy ('absolute', 'relative', 'local')
        scale: If scale=True, then scaling of positive values to range [0,1]
        penalty: Set values below threshold to value specified 
        binarize: Binarizes final matrix (positive: 1; otherwise: 0)
        Note: Binarization is applied last (overriding other settings)
        

    Returns:
        S_thresh: Thresholded matrix
    """
    if np.min(S)<0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N*M
    
    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0
        
    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N,M])  
            
    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N,M])   
        num_cells_row_below_thresh = int(np.round(M*(1-thresh_rel_row)))  
        for n in range(N):
            row = S[n,:]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n,:] = (row>=thresh_abs)
        S_binary_col = np.zeros([N,M])
        num_cells_col_below_thresh = int(np.round(N*(1-thresh_rel_col)))  
        for m in range(M):
            col = S[:,m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:,m] = (col>=thresh_abs)
        S_thresh =  S * S_binary_row * S_binary_col
        
    if scale: 
        cell_val_zero = np.where(S_thresh==0)
        cell_val_pos = np.where(S_thresh>0)
        if len(cell_val_pos[0])==0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])  
        max_value = np.max(S_thresh)
        #print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value) , (max_value -  min_value)) 
            if len(cell_val_zero[0])>0:
                S_thresh[cell_val_zero] = penalty   
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')    
            
    if binarize:
        S_thresh[S_thresh > 0] = 1 
        S_thresh[S_thresh < 0] = 0
    return S_thresh


def compute_SM_from_filename(fn_wav, L=21, H=5, L_smooth=16, tempo_rel_set=np.array([1]), shift_set=np.array([0]), 
                           strategy = 'relative', scale=1, thresh=0.15, penalty=0, binarize=0):  
    """Compute self similarity matrix for specified audio file
    
    Notebook: C4S2_SSM-Thresholding.ipynb
    
    Args: 
        fn_wav: Path and filename of wav file
        L, H: Parameters for computing smoothed chroma features
        L_smooth, tempo_rel_set, shift_set: Parameters for computing SSM
        strategy, scale, thresh, penalty, binarize: Parameters used thresholding SSM

    Returns: 
        x, x_duration: Audio signal and its duration (seconds) 
        X, Fs_feature: Feature sequence and feature rate
        S_thresh, I: SSM and index matrix
    """    
    # Waveform    
    Fs = 22050
    x, Fs = librosa.load(fn_wav, sr=Fs) 
    x_duration = (x.shape[0])/Fs

    # Chroma Feature Sequence and SSM (10 Hz)
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
    Fs_C = Fs/2205

    # Chroma Feature Sequence and SSM
    X, Fs_feature = smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
    X = normalize_feature_sequence(X, norm='2', threshold=0.001)

    # Compute SSM   
    S, I = compute_SM_TI(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    S_thresh = threshold_matrix(S, thresh=thresh, strategy=strategy, 
                                          scale=scale, penalty=penalty, binarize=binarize)
    return x, x_duration, X, Fs_feature, S_thresh, I

import os

import scipy.stats
from scipy.io import loadmat
import numpy as np
import pandas as pd

N_PITCH_CLS = 12 # {C, C#, ..., Bb, B}

def get_event_seq(piece_csv, seq_col_name='ENCODING'):
  '''
  Extracts the event sequence from a piece of music (stored in .csv file).
  NOTE: You should modify this function if you use different formats.

  Parameters:
    piece_csv (str): path to the piece's .csv file.
    seq_col_name (str): name of the column containing event encodings.

  Returns:
    list: the event sequence of the piece.
  '''
  df = pd.read_csv(piece_csv, encoding='utf-8')
  return df[seq_col_name].astype('int32').tolist()


def get_chord_sequence(ev_seq, chord_evs):
  '''
  Extracts the chord sequence (in string representation) from the input piece.
  NOTE: This function is vocabulary-dependent, 
        you should implement a new one if a different vocab is used. 

  Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    chord_evs (dict of lists): [key] type of chord-related event --> [value] encodings belonging to the type.

  Returns:
    list of lists: The chord sequence of the input piece, each element (a list) being the representation of a single chord.
  '''
  # extract chord-related tokens
  ev_seq = [
    x for x in ev_seq if any(x in chord_evs[typ] for typ in chord_evs.keys())
  ]

  # remove grammar errors in sequence (vocabulary-dependent)
  legal_seq = []
  cnt = 0
  for i, ev in enumerate(ev_seq):
    cnt += 1
    if ev in chord_evs['Chord-Slash'] and cnt == 3:
      cnt = 0
      legal_seq.extend(ev_seq[i-2:i+1])
  
  ev_seq = legal_seq
  assert not len(ev_seq) % 3
  chords = []
  for i in range(0, len(ev_seq), 3):
    chords.append( ev_seq[i:i+3] )

  return chords

def compute_histogram_entropy(hist):
  ''' 
  Computes the entropy (log base 2) of a normalised histogram.

  Parameters:
    hist (ndarray): input pitch (or duration) histogram, should be normalised.

  Returns:
    float: entropy (log base 2) of the histogram.
  '''
  return scipy.stats.entropy(hist) / np.log(2)


def get_pitch_histogram(ev_seq, pitch_evs=range(128), verbose=False):
  '''
  Computes the pitch-class histogram from an event sequence.

  Parameters:
    ev_seq (list): a piece of music in event sequence representation.
    pitch_evs (list): encoding IDs of ``Note-On`` events, should be sorted in increasing order by pitches.
    verbose (bool): whether to print msg. when ev_seq has no notes.

  Returns:
    ndarray: the resulting pitch-class histogram.
  '''
  ev_seq = [x for x in ev_seq if x in pitch_evs]

  if not len(ev_seq):
    if verbose:
      print ('[Info] The sequence contains no notes.')
    return None

  # compress sequence to pitch classes & get normalised counts
  ev_seq = pd.Series(ev_seq) % N_PITCH_CLS
  ev_hist = ev_seq.value_counts(normalize=True)

  # make the final histogram
  hist = np.zeros( (N_PITCH_CLS,) )
  for i in range(N_PITCH_CLS):
    if i in ev_hist.index:
      hist[i] = ev_hist.loc[i]

  return hist

def get_onset_xor_distance(seq_a, seq_b, bar_ev_id, pos_evs, pitch_evs=range(128)):
  '''
  Computes the XOR distance of onset positions between a pair of bars.
  
  Parameters:
    seq_a, seq_b (list): event sequence of a bar of music.
      IMPORTANT: for this implementation, a ``Note-Position`` event must appear before the associated ``Note-On``.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    pos_evs (list): encoding IDs of ``Note-Position`` events, vocabulary-dependent.
    pitch_evs (list): encoding IDs of ``Note-On`` events.

  Returns:
    float: 0~1, the XOR distance between the 2 bars' (seq_a, seq_b) binary vectors of onsets.
  '''
  # sanity checks
  assert seq_a[0] == bar_ev_id and seq_b[0] == bar_ev_id
  assert seq_a.count(bar_ev_id) == 1 and seq_b.count(bar_ev_id) == 1

  # compute binary onset vectors
  n_pos = len(pos_evs)
  def make_onset_vec(seq):
    cur_pos = -1
    onset_vec = np.zeros((n_pos,))
    for ev in seq:
      if ev in pos_evs:
        cur_pos = ev - pos_evs[0]
      if ev in pitch_evs:
        onset_vec[cur_pos] = 1
    return onset_vec
  a_onsets, b_onsets = make_onset_vec(seq_a), make_onset_vec(seq_b)

  # compute XOR distance
  dist = np.sum( np.abs(a_onsets - b_onsets) ) / n_pos
  return dist

def get_bars_crop(ev_seq, start_bar, end_bar, bar_ev_id, verbose=False):
  '''
  Returns the designated crop (bars) of the input piece.

  Parameter:
    ev_seq (list): a piece of music in event sequence representation.
    start_bar (int): the starting bar of the crop.
    end_bar (int): the ending bar (inclusive) of the crop.
    bar_ev_id (int): encoding ID of the ``Bar`` event, vocabulary-dependent.
    verbose (bool): whether to print messages when unexpected operations happen.

  Returns:
    list: a cropped segment of music consisting of (end_bar - start_bar + 1) bars.
  '''
  if start_bar < 0 or end_bar < 0:
    raise ValueError('Invalid start_bar: {}, or end_bar: {}.'.format(start_bar, end_bar))

  # get the indices of ``Bar`` events
  ev_seq = np.array(ev_seq)
  bar_markers = np.where(ev_seq == bar_ev_id)[0]

  if start_bar > len(bar_markers) - 1:
    raise ValueError('start_bar: {} beyond end of piece.'.format(start_bar))

  if end_bar < len(bar_markers) - 1:
    cropped_seq = ev_seq[ bar_markers[start_bar] : bar_markers[end_bar + 1] ]
  else:
    if verbose:
      print (
        '[Info] end_bar: {} beyond or equal the end of the input piece; only the last {} bars are returned.'.format(
          end_bar, len(bar_markers) - start_bar
        ))
    cropped_seq = ev_seq[ bar_markers[start_bar] : ]

  return cropped_seq.tolist()

def read_fitness_mat(fitness_mat_file):
  '''
  Reads and returns (as an ndarray) a fitness scape plot as a center-duration matrix.

  Parameters:
    fitness_mat_file (str): path to the file containing fitness scape plot.
      Accepted formats: .mat (MATLAB data), .npy (ndarray)

  Returns:
    ndarray: the fitness scapeplot encoded as a center-duration matrix.
  '''
  ext = os.path.splitext(fitness_mat_file)[-1].lower()

  if ext == '.npy':
    f_mat = np.load(fitness_mat_file)
  elif ext == '.mat':
    mat_dict = loadmat(fitness_mat_file)
    f_mat = mat_dict['fitness_info'][0, 0][0]
    f_mat[ np.isnan(f_mat) ] = 0.0
  else:
    raise ValueError('Unsupported fitness scape plot format: {}'.format(ext))

  for slen in range(f_mat.shape[0]):
    f_mat[slen] = np.roll(f_mat[slen], slen // 2)

  return f_mat


######################################################
# DEPRECATED FUNCTIONS
######################################################
# def read_fitness_mat(mat_file):
#   '''
#   Reads and returns (as an ndarray) a fitness scape plot stored in MATLAB .mat format.

#   Parameters:
#     mat_file (str): path to the .mat file containing fitness scape plot. (computed by ``run_matlab_scapeplot.py``).

#   Returns:
#     ndarray: the fitness scapeplot manipulable in Python.
#   '''
#   mat_dict = loadmat(mat_file)
#   f_mat = mat_dict['fitness_info'][0, 0][0]
#   f_mat[ np.isnan(f_mat) ] = 0.0

#   for slen in range(f_mat.shape[0]):
#     f_mat[slen] = np.roll(f_mat[slen], slen // 2)

#   return f_mat