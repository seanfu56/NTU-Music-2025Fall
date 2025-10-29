import torchaudio
import numpy as np
from scipy.signal import savgol_filter
import librosa
import torch
import torchaudio
import scipy.signal as signal
from torchaudio import transforms as T
import torch
import torchaudio
import librosa
import numpy as np
def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256):
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma

def melody_similarity(target_path, reference_path):
    gt_melody = extract_melody_one_hot(target_path)      
    ref_melody = extract_melody_one_hot(reference_path)
    min_len_melody = min(gt_melody.shape[1], ref_melody.shape[1])
    matches = ((gt_melody[:, :min_len_melody] == ref_melody[:, :min_len_melody]) & (gt_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    return accuracy
    