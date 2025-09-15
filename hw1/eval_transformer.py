#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


# =============================
# Audio + features
# =============================

TARGET_SR = 16000
N_FFT = 1024
HOP = 160  # 10ms
WIN = 400  # 25ms
N_MELS = 128


def load_audio(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP,
    win_length=WIN,
    n_mels=N_MELS,
    power=2.0,
)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    x = mel_spec(wav)
    x = amp_to_db(x)
    m = x.mean()
    s = x.std()
    x = (x - m) / (s + 1e-5)
    return x


def slide_chunks(wav: torch.Tensor, chunk_sec: float = 3.0, overlap: float = 0.5, sr: int = TARGET_SR) -> List[torch.Tensor]:
    chunk_len = int(chunk_sec * sr)
    if wav.numel() <= chunk_len:
        # pad-repeat to min length
        reps = (chunk_len + wav.numel() - 1) // wav.numel()
        wav = wav.repeat(reps)[:chunk_len]
        return [wav]
    step = max(1, int(chunk_len * (1 - overlap)))
    starts = list(range(0, max(1, wav.numel() - chunk_len + 1), step))
    if starts[-1] != wav.numel() - chunk_len:
        starts.append(wav.numel() - chunk_len)
    return [wav[s : s + chunk_len] for s in starts]


def _load_demucs_model(device: torch.device):
    """Lazy-load a Demucs model for on-the-fly vocal separation.
    Requires `demucs` package installed. Returns model or None on failure.
    """
    try:
        from demucs.pretrained import get_model  # type: ignore
        model = get_model(name="htdemucs")
        model.to(device)
        model.eval()
        return model
    except Exception:
        try:
            model = torch.hub.load("facebookresearch/demucs:main", "htdemucs")  # type: ignore
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"[warn] Could not load Demucs for separation: {e}")
            return None


def separate_vocals_in_memory(wav: torch.Tensor, sr: int, device: torch.device) -> torch.Tensor:
    """Separate vocals from a mono waveform using Demucs, returning a mono vocals track.
    Does not write any files; runs fully in memory. If Demucs is unavailable, returns input.
    """
    model = _load_demucs_model(device)
    if model is None:
        return wav
    DEMUCS_SR = 44100
    x = wav.unsqueeze(0)  # [1, T]
    if sr != DEMUCS_SR:
        x = torchaudio.functional.resample(x, sr, DEMUCS_SR)
    # Demucs expects [B, C, T], stereo preferred
    x = torch.vstack([x, x]).unsqueeze(0).to(device)  # [1, 2, T]
    try:
        from demucs.apply import apply_model  # type: ignore
        with torch.no_grad():
            out = apply_model(model, x, split=True, overlap=0.25, device=device)
        # out: [B, sources, C, T]
        out = out.squeeze(0)  # [sources, C, T]
        src_names = getattr(model, "sources", None)
        v_idx = (src_names.index("vocals") if isinstance(src_names, list) and "vocals" in src_names else -1)
        vocals = out[v_idx] if v_idx >= 0 else out[-1]  # [C, T]
        v_mono = vocals.mean(dim=0)  # [T]
        if DEMUCS_SR != TARGET_SR:
            v_mono = torchaudio.functional.resample(v_mono.unsqueeze(0), DEMUCS_SR, TARGET_SR).squeeze(0)
        return v_mono.cpu()
    except Exception as e:
        print(f"[warn] Demucs separation failed, fallback to mix: {e}")
        return wav


# =============================
# Model (same as train_transformer)
# =============================


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.use_proj = in_ch != out_ch
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if self.use_proj else None
        self.proj_bn = nn.BatchNorm2d(out_ch) if self.use_proj else None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        do_pool = out.size(-2) >= 2
        if do_pool:
            out = self.pool(out)
        if self.use_proj and self.proj is not None:
            identity = self.proj(identity)
            identity = self.proj_bn(identity)  # type: ignore[arg-type]
        if do_pool:
            identity = self.pool(identity)
        out = out + identity
        out = self.act(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class CNNTransformer(nn.Module):
    def __init__(self, n_mels: int, n_classes: int,
                 depth: int = 6, base_channels: int = 64,
                 d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_c = 1
        out_c = base_channels
        for _ in range(max(1, int(depth))):
            blocks.append(ResidualConvBlock(in_c, out_c))
            in_c = out_c
            out_c *= 2
        self.blocks = nn.ModuleList(blocks)

        self.proj = None
        self._proj_in_dim = None
        self.d_model = int(d_model)
        self.posenc = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(n_heads), dim_feedforward=int(ff_dim),
            dropout=float(dropout), batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, n_classes),
        )

    def _ensure_proj(self, m_freq_red: int, channels: int, device: torch.device):
        in_dim = channels * m_freq_red
        if self._proj_in_dim != in_dim:
            self._proj_in_dim = in_dim
            self.proj = nn.Linear(in_dim, self.d_model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        B, C, Mred, T = x.shape
        self._ensure_proj(Mred, C, x.device)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * Mred)
        x = self.proj(x)
        x = self.posenc(x)
        x = self.encoder(x)
        mean_t = x.mean(dim=1)
        std_t = x.std(dim=1)
        feat = torch.cat([mean_t, std_t], dim=1)
        logits = self.head(feat)
        return logits


# =============================
# Eval helpers
# =============================

@torch.no_grad()
def predict_track_from_wav(
    model: nn.Module,
    wav: torch.Tensor,
    device: torch.device,
    *,
    chunk_sec: float = 4.0,
    overlap: float = 0.5,
    batch_size: int = 96,
    max_segments: int | None = 64,
    agg: str = "mean",
) -> torch.Tensor:
    chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
    if max_segments is not None and max_segments > 0 and len(chunks) > max_segments:
        idxs = np.linspace(0, len(chunks) - 1, num=max_segments, dtype=int).tolist()
        chunks = [chunks[i] for i in idxs]
    logps: List[torch.Tensor] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]
        x = torch.stack(feats, dim=0).to(device)
        logits = model(x)
        logp = F.log_softmax(logits, dim=-1).detach().cpu()
        logps.append(logp)
    logprobs = torch.cat(logps, dim=0) if logps else torch.zeros(1, 1)
    if agg == "median":
        return torch.median(logprobs, dim=0).values
    return logprobs.mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN+Transformer on test set (vocals-only mapping)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt from training")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--dataset_subdir", type=str, default="artist20_vocals", help="Root of original/test data")
    parser.add_argument("--output", type=str, default=None, help="Output predictions JSON path (default next to ckpt)")
    # Vocals separation
    parser.add_argument("--use_vocals_only", action="store_true",
                        help="On-the-fly separate vocals from mix (in-memory) for inference")
    # Model shape (must match training)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--tx_layers", type=int, default=6)
    parser.add_argument("--tx_heads", type=int, default=8)
    parser.add_argument("--tx_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Eval controls
    parser.add_argument("--chunk_sec", type=float, default=4.0)
    parser.add_argument("--eval_overlap", type=float, default=0.5)
    parser.add_argument("--eval_batch_size", type=int, default=96)
    parser.add_argument("--eval_max_segments", type=int, default=64)
    parser.add_argument("--eval_agg", type=str, default="mean", choices=["mean", "median"])
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_map: Dict[str, int] = ckpt["label_map"] if "label_map" in ckpt else {}
    if not label_map:
        raise SystemExit("Checkpoint missing label_map. Please re-train with newer script or provide labels.")

    inv_label = {v: k for k, v in label_map.items()}
    n_classes = len(inv_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformer(
        n_mels=N_MELS,
        n_classes=n_classes,
        depth=args.depth,
        base_channels=args.base_channels,
        d_model=args.d_model,
        n_heads=args.tx_heads,
        n_layers=args.tx_layers,
        ff_dim=args.tx_ff,
        dropout=args.dropout,
    ).to(device)
    sd = ckpt.get("model", {})
    if "proj.weight" in sd:
        w = sd["proj.weight"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        model.proj = nn.Linear(in_f, out_f).to(device)
        model._proj_in_dim = in_f
    model.load_state_dict(sd, strict=True)
    model.eval()

    data_root = Path(args.data_root)
    dataset_dir = data_root / args.dataset_subdir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset dir not found: {dataset_dir}")
    test_dir = dataset_dir / "test"
    test_files = sorted(list(test_dir.glob("*.wav")) + list(test_dir.glob("*.mp3")))
    if not test_files:
        raise SystemExit(f"No test audio found under {test_dir}")

    pred: Dict[str, List[str]] = {}
    for p in tqdm(test_files, desc="Test", ncols=80):
        wav = load_audio(str(p))
        if bool(args.use_vocals_only):
            wav = separate_vocals_in_memory(wav, TARGET_SR, device)
        logp = predict_track_from_wav(
            model,
            wav,
            device,
            chunk_sec=args.chunk_sec,
            overlap=args.eval_overlap,
            batch_size=args.eval_batch_size,
            max_segments=(None if args.eval_max_segments <= 0 else int(args.eval_max_segments)),
            agg=str(args.eval_agg),
        )
        topk = torch.topk(logp, k=min(3, n_classes)).indices.tolist()
        pred[p.stem] = [inv_label[i] for i in topk]

    out_path = Path(args.output) if args.output else (ckpt_path.parent / "test_pred.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=2)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
