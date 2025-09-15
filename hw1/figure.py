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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# =============================
# Audio + features (match training)
# =============================

TARGET_SR = 16000
N_FFT = 1024
HOP = 160
WIN = 400
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


def slide_chunks(wav: torch.Tensor, chunk_sec: float = 4.0, overlap: float = 0.5, sr: int = TARGET_SR) -> List[torch.Tensor]:
    chunk_len = int(chunk_sec * sr)
    if wav.numel() <= chunk_len:
        reps = (chunk_len + wav.numel() - 1) // wav.numel()
        wav = wav.repeat(reps)[:chunk_len]
        return [wav]
    step = max(1, int(chunk_len * (1 - overlap)))
    starts = list(range(0, max(1, wav.numel() - chunk_len + 1), step))
    if starts[-1] != wav.numel() - chunk_len:
        starts.append(wav.numel() - chunk_len)
    return [wav[s : s + chunk_len] for s in starts]


def try_vocals_path(orig_path: str, dataset_root: Path | None, vocals_root: Path | None) -> str:
    if vocals_root is None or dataset_root is None:
        return orig_path
    try:
        rel = os.path.relpath(orig_path, start=str(dataset_root))
    except Exception:
        return orig_path
    base = vocals_root / rel
    if base.exists():
        bw = base.with_suffix('.wav')
        return str(bw if bw.exists() else base)
    stem = Path(rel).with_suffix("")
    for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        cand = vocals_root / (str(stem) + ext)
        if cand.exists():
            return str(cand)
    stem_name = Path(orig_path).stem
    matches = list(vocals_root.glob(f"**/{stem_name}/vocals.*"))
    if matches:
        return str(matches[0])
    return orig_path


# =============================
# Model (copy of training)
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
        feat = torch.cat([mean_t, std_t], dim=1)  # [B, 2D]
        logits = self.head(feat)
        return logits


# =============================
# Data + labels I/O
# =============================


def _artist_from_path(p: Path) -> str:
    parts = list(p.parts)
    if "train_val" in parts:
        try:
            idx = parts.index("train_val")
            return parts[idx + 1]
        except Exception:
            pass
    return p.parent.parent.name


def load_split_list(dataset_dir: Path, json_name: str) -> Tuple[List[str], List[str]]:
    with open(dataset_dir / json_name, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    files, labels = [], []
    for rel in rel_list:
        p = dataset_dir / rel.lstrip("./")
        if not p.exists():
            continue
        files.append(str(p))
        labels.append(_artist_from_path(p))
    return files, labels


# =============================
# Embedding extraction
# =============================


@torch.no_grad()
def extract_track_embedding(
    model: nn.Module,
    path: str,
    device: torch.device,
    *,
    chunk_sec: float = 4.0,
    overlap: float = 0.5,
    batch_size: int = 96,
    max_segments: int | None = 64,
    hook_layer: nn.Module | None = None,
) -> np.ndarray:
    # Capture activations after the first linear+ReLU in head
    captured: List[torch.Tensor] = []

    def _hook(_m, _in, out):
        captured.append(out.detach().cpu())

    handle = None
    if hook_layer is not None:
        handle = hook_layer.register_forward_hook(_hook)

    wav = load_audio(path)
    chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
    if max_segments is not None and max_segments > 0 and len(chunks) > max_segments:
        idxs = np.linspace(0, len(chunks) - 1, num=max_segments, dtype=int).tolist()
        chunks = [chunks[i] for i in idxs]

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]
        x = torch.stack(feats, dim=0).to(device)
        _ = model(x)  # logits ignored; hook collects embeddings

    if handle is not None:
        handle.remove()

    if not captured:
        return np.zeros((1,), dtype=np.float32)
    E = torch.cat(captured, dim=0)  # [Nseg, D]
    emb = E.mean(dim=0).numpy()  # [D]
    return emb


def tsne_plot(points: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    scaler = StandardScaler()
    X = scaler.fit_transform(points)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    Z = tsne.fit_transform(X)

    classes = sorted(set(labels))
    cmap = plt.get_cmap('tab20')
    color_map = {c: cmap(i % 20) for i, c in enumerate(classes)}

    plt.figure(figsize=(9, 7))
    for c in classes:
        idxs = [i for i, lab in enumerate(labels) if lab == c]
        plt.scatter(Z[idxs, 0], Z[idxs, 1], s=10, alpha=0.7, label=c, c=[color_map[c]])
    plt.title(title)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def tsne_plot_joint(
    points_tr: np.ndarray,
    labels_tr: List[str],
    points_va: np.ndarray,
    labels_va: List[str],
    title: str,
    out_path: Path,
) -> None:
    # Fit t-SNE on concatenated set to share the same space
    X = np.concatenate([points_tr, points_va], axis=0)
    L = labels_tr + labels_va
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    Z = tsne.fit_transform(Xn)
    Ztr = Z[: len(points_tr)]
    Zva = Z[len(points_tr) :]

    classes = sorted(set(L))
    cmap = plt.get_cmap('tab20')
    color_map = {c: cmap(i % 20) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(9, 7))
    # Plot per class: train filled, val hollow
    for c in classes:
        idx_tr = [i for i, lab in enumerate(labels_tr) if lab == c]
        idx_va = [i for i, lab in enumerate(labels_va) if lab == c]
        if idx_tr:
            ax.scatter(Ztr[idx_tr, 0], Ztr[idx_tr, 1], s=10, alpha=0.7, c=[color_map[c]], linewidths=0.2)
        if idx_va:
            ax.scatter(Zva[idx_va, 0], Zva[idx_va, 1], s=18, facecolors='none', edgecolors=[color_map[c]], linewidths=0.6)

    # Legends: class-color legend (outside, right) + dataset-style legend (inside)
    from matplotlib.lines import Line2D
    class_handles = [
        Line2D([0], [0], marker='o', color='w', label=c, markerfacecolor=color_map[c], markeredgecolor=color_map[c], markersize=6)
        for c in classes
    ]
    class_legend = ax.legend(handles=class_handles, title='Artist', fontsize=6, title_fontsize=7,
                             bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.add_artist(class_legend)

    style_handles = [
        Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='gray', markeredgecolor='gray', markersize=6),
        Line2D([0], [0], marker='o', color='gray', label='Val', markerfacecolor='white', markeredgecolor='gray', markersize=7),
    ]
    ax.legend(handles=style_handles, loc='lower right', fontsize=7, title='Set', title_fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="t-SNE of transformer embeddings for train/val")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt from training")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--dataset_subdir", type=str, default="artist20_vocals")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    # Vocals mapping (optional)
    parser.add_argument("--use_vocals_only", action="store_true")
    parser.add_argument("--vocals_root", type=str, default=None)
    # Model hyperparams (must match training)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--tx_layers", type=int, default=6)
    parser.add_argument("--tx_heads", type=int, default=8)
    parser.add_argument("--tx_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Embedding extraction controls
    parser.add_argument("--chunk_sec", type=float, default=4.0)
    parser.add_argument("--eval_overlap", type=float, default=0.5)
    parser.add_argument("--eval_batch_size", type=int, default=96)
    parser.add_argument("--eval_max_segments", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(Path(args.ckpt), map_location="cpu")
    label_map: Dict[str, int] = ckpt["label_map"] if "label_map" in ckpt else {}
    if not label_map:
        raise SystemExit("Checkpoint missing label_map. Cannot label embeddings.")
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
    # If checkpoint contains a saved projection layer shape, precreate it
    sd = ckpt.get("model", {})
    if "proj.weight" in sd:
        w = sd["proj.weight"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        model.proj = nn.Linear(in_f, out_f).to(device)
        model._proj_in_dim = in_f
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Hook layer: output after ReLU in head -> index 1
    hook_layer = model.head[1]

    data_root = Path(args.data_root)
    dataset_dir = data_root / args.dataset_subdir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, train_names = load_split_list(dataset_dir, "train.json")
    val_files, val_names = load_split_list(dataset_dir, "val.json")

    vocals_root = Path(args.vocals_root) if args.vocals_root else None

    # Extract embeddings
    def _proc(file_list: List[str], names: List[str], tag: str) -> Tuple[np.ndarray, List[str]]:
        embs: List[np.ndarray] = []
        labs: List[str] = []
        for p, lab in tqdm(list(zip(file_list, names)), desc=f"Emb {tag}", ncols=80):
            path = p
            if bool(args.use_vocals_only):
                path = try_vocals_path(path, dataset_dir, vocals_root)
            e = extract_track_embedding(
                model,
                path,
                device,
                chunk_sec=args.chunk_sec,
                overlap=args.eval_overlap,
                batch_size=args.eval_batch_size,
                max_segments=(None if args.eval_max_segments <= 0 else int(args.eval_max_segments)),
                hook_layer=hook_layer,
            )
            if e.size > 1:
                embs.append(e)
                labs.append(lab)
        if not embs:
            raise SystemExit(f"No embeddings for {tag} set.")
        return np.stack(embs, axis=0), labs

    Xtr, Ltr = _proc(train_files, train_names, "train")
    Xva, Lva = _proc(val_files, val_names, "val")

    # Save raw embeddings
    np.save(out_dir / "emb_train.npy", Xtr)
    np.save(out_dir / "emb_val.npy", Xva)
    with open(out_dir / "emb_labels.json", "w", encoding="utf-8") as f:
        json.dump({"train": Ltr, "val": Lva}, f, ensure_ascii=False, indent=2)

    # Plot t-SNE for train and val separately
    tsne_plot(Xtr, Ltr, title="t-SNE (Train)", out_path=out_dir / "tsne_train.png")
    tsne_plot(Xva, Lva, title="t-SNE (Val)", out_path=out_dir / "tsne_val.png")
    # Joint plot in one figure with different marker styles
    tsne_plot_joint(Xtr, Ltr, Xva, Lva, title="t-SNE (Train + Val)", out_path=out_dir / "tsne_train_val.png")
    print(f"Wrote embeddings and plots to {out_dir}")


if __name__ == "__main__":
    main()
