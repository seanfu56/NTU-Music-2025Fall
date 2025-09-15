#!/usr/bin/env python3
import argparse
import json
import os
import time
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


# =============================
# Audio and feature pipeline
# =============================

TARGET_SR = 16000
N_FFT = 1024
HOP = 160  # 10ms
WIN = 400  # 25ms
N_MELS = 128


def load_audio(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [T]


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
    return x  # [M, T]


def slice_random_chunk(wav: torch.Tensor, chunk_sec: float = 3.0, sr: int = TARGET_SR) -> torch.Tensor:
    chunk_len = int(chunk_sec * sr)
    if wav.numel() < chunk_len:
        reps = (chunk_len + wav.numel() - 1) // wav.numel()
        wav = wav.repeat(reps)[:chunk_len]
        return wav
    start = int(torch.randint(0, wav.numel() - chunk_len + 1, (1,)).item())
    return wav[start : start + chunk_len]


def slide_chunks(wav: torch.Tensor, chunk_sec: float = 3.0, overlap: float = 0.5, sr: int = TARGET_SR) -> List[torch.Tensor]:
    chunk_len = int(chunk_sec * sr)
    if wav.numel() <= chunk_len:
        return [slice_random_chunk(wav, chunk_sec, sr)]
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
        return str(base.with_suffix('.wav')) if base.with_suffix('.wav').exists() else str(base)
    stem = Path(rel).with_suffix("")
    for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        cand = vocals_root / (str(stem) + ext)
        if cand.exists():
            return str(cand)
    return orig_path


# =============================
# Dataset
# =============================

class TrainChunkDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], chunk_sec: float = 3.0,
                 augment: bool = True, spec_time_mask: int = 30, spec_freq_mask: int = 24,
                 spec_num_time_masks: int = 2, spec_num_freq_masks: int = 2,
                 dataset_root: Path | None = None, vocals_root: Path | None = None, prefer_vocals: bool = False):
        self.files = files
        self.labels = labels
        self.chunk_sec = chunk_sec
        self.augment = augment
        self.spec_time_mask = int(spec_time_mask)
        self.spec_freq_mask = int(spec_freq_mask)
        self.spec_num_time_masks = int(spec_num_time_masks)
        self.spec_num_freq_masks = int(spec_num_freq_masks)
        self.dataset_root = dataset_root
        self.vocals_root = vocals_root
        self.prefer_vocals = prefer_vocals

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        if self.prefer_vocals:
            path = try_vocals_path(path, self.dataset_root, self.vocals_root)
        y = self.labels[idx]
        wav = load_audio(path)
        chunk = slice_random_chunk(wav, self.chunk_sec)
        feat = wav_to_logmel(chunk)  # [M, T]
        if self.augment:
            for _ in range(max(0, self.spec_num_freq_masks)):
                feat = torchaudio.transforms.FrequencyMasking(self.spec_freq_mask)(feat)
            for _ in range(max(0, self.spec_num_time_masks)):
                feat = torchaudio.transforms.TimeMasking(self.spec_time_mask)(feat)
        return feat.unsqueeze(0), y  # [1, M, T]


# =============================
# Model: CNN + Transformer Encoder
# =============================

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # pool over frequency only
        self.use_proj = in_ch != out_ch
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if self.use_proj else None
        self.proj_bn = nn.BatchNorm2d(out_ch) if self.use_proj else None

    def forward(self, x):  # [B,C,M,T]
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
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class CNNTransformer(nn.Module):
    def __init__(self, n_mels: int, n_classes: int,
                 depth: int = 4, base_channels: int = 64,
                 d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 3, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        # CNN frontend
        blocks = []
        in_c = 1
        out_c = base_channels
        for _ in range(max(1, int(depth))):
            blocks.append(ResidualConvBlock(in_c, out_c))
            in_c = out_c
            out_c *= 2
        self.blocks = nn.ModuleList(blocks)

        # Project CNN feature (C*M') to d_model per time step
        self.proj = None
        self._proj_in_dim = None  # determined at runtime after seeing M'
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,1,M,T]
        for blk in self.blocks:
            x = blk(x)
        # x: [B, C, M', T]
        B, C, Mred, T = x.shape
        self._ensure_proj(Mred, C, x.device)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, M']
        x = x.view(B, T, C * Mred)  # [B, T, C*M']
        x = self.proj(x)  # [B,T,D]
        x = self.posenc(x)
        x = self.encoder(x)  # [B,T,D]
        mean_t = x.mean(dim=1)
        std_t = x.std(dim=1)
        feat = torch.cat([mean_t, std_t], dim=1)  # [B,2D]
        logits = self.head(feat)
        return logits


# =============================
# Utils: splits and labels
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
    full_paths: List[str] = []
    labels: List[str] = []
    for rel in rel_list:
        p = dataset_dir / rel.lstrip("./")
        if not p.exists():
            print(f"[warn] Missing file listed in {json_name}: {rel}")
            continue
        # Prefer .wav counterpart if exists (vocals set)
        pwav = p.with_suffix('.wav')
        full_paths.append(str(pwav if pwav.exists() else p))
        labels.append(_artist_from_path(p))
    return full_paths, labels


def build_label_map(train_labels: List[str], val_labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(train_labels + val_labels))
    return {name: idx for idx, name in enumerate(uniq)}


# =============================
# Training / Evaluation
# =============================

@torch.no_grad()
def evaluate_trackwise(
    model: nn.Module,
    files: List[str],
    labels: List[int],
    device: torch.device,
    *,
    chunk_sec: float = 3.0,
    overlap: float = 0.5,
    batch_size: int = 64,
    max_segments: int | None = None,
    agg: str = "mean",
) -> Tuple[float, float, float]:
    model.eval()
    all_top1, all_top3, val_losses = [], [], []
    for path, y in tqdm(list(zip(files, labels)), desc="Eval", ncols=80):
        wav = load_audio(path)
        chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
        if max_segments is not None and max_segments > 0 and len(chunks) > max_segments:
            idxs = np.linspace(0, len(chunks) - 1, num=max_segments, dtype=int).tolist()
            chunks = [chunks[i] for i in idxs]
        logprob_list = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]
            x = torch.stack(feats, dim=0).to(device)
            logits = model(x)
            logp = F.log_softmax(logits, dim=-1).detach().cpu()
            logprob_list.append(logp)
        logprobs = torch.cat(logprob_list, dim=0)  # [Nseg, C]
        if agg == "median":
            logprob_mean = torch.median(logprobs, dim=0).values
        else:
            logprob_mean = logprobs.mean(dim=0)
        top3 = torch.topk(logprob_mean, k=3).indices.tolist()
        pred = top3[0]
        all_top1.append(1 if pred == y else 0)
        all_top3.append(1 if y in top3 else 0)
        val_losses.append(float(-logprob_mean[int(y)]))
    top1 = float(np.mean(all_top1)) if all_top1 else 0.0
    top3 = float(np.mean(all_top3)) if all_top3 else 0.0
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    return top1, top3, val_loss


def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return -(targets * logp).sum(dim=-1).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    mixup_p: float = 1.0,
    clip_grad_norm: float = 0.0,
    amp: bool = False,
) -> Tuple[float, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total, n, correct = 0.0, 0, 0
    for feats, y in tqdm(loader, desc="Train", ncols=80):
        feats = feats.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        do_mix = float(mixup_alpha) > 0.0 and float(mixup_p) > 0.0 and (torch.rand(()) < float(mixup_p))
        with torch.cuda.amp.autocast(enabled=amp):
            if do_mix:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(feats.size(0), device=feats.device)
                feats = lam * feats + (1.0 - lam) * feats[perm]
                logits = model(feats)
                num_classes = logits.size(-1)
                t1 = _one_hot(y, num_classes)
                t2 = _one_hot(y[perm], num_classes)
                targets = lam * t1 + (1.0 - lam) * t2
                if label_smoothing and label_smoothing > 0.0:
                    targets = (1.0 - label_smoothing) * targets + label_smoothing / num_classes
                loss = _soft_ce(logits, targets)
            else:
                logits = model(feats)
                loss = F.cross_entropy(logits, y, label_smoothing=float(label_smoothing))
        scaler.scale(loss).backward()
        if clip_grad_norm and clip_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * y.size(0)
        n += y.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == y).sum().item()
    return total / max(1, n), correct / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="CNN+Transformer singer classifier (PyTorch)")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--dataset_subdir", type=str, default="artist20_vocals")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_sec", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for cross-entropy (e.g., 0.05)")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup alpha (0 disables mixup)")
    parser.add_argument("--mixup_p", type=float, default=1.0,
                        help="Probability to apply mixup per batch")
    parser.add_argument("--clip_grad_norm", type=float, default=0.0,
                        help="Gradient clipping max-norm (0 disables)")
    # Model
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--tx_layers", type=int, default=6)
    parser.add_argument("--tx_heads", type=int, default=8)
    parser.add_argument("--tx_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.add_argument("--spec_time_mask", type=int, default=40)
    parser.add_argument("--spec_freq_mask", type=int, default=32)
    parser.add_argument("--spec_num_time_masks", type=int, default=4)
    parser.add_argument("--spec_num_freq_masks", type=int, default=2)
    # Eval controls
    parser.add_argument("--eval_overlap", type=float, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=96)
    parser.add_argument("--eval_max_segments", type=int, default=64)
    parser.add_argument("--eval_agg", type=str, default="mean", choices=["mean", "median"],
                        help="Chunk aggregation for eval/test")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--balance_sampler", action="store_true",
                        help="Use class-balanced WeightedRandomSampler for training")
    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    # AMP
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ntu-music-singer-cnn")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    dataset_dir = data_root / args.dataset_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb and WANDB_AVAILABLE:
        base_config = vars(args).copy()
        # Remove objects not serializable
        for k in list(base_config.keys()):
            if isinstance(base_config[k], (Path,)):
                base_config[k] = str(base_config[k])
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_run_name, config=base_config)

    train_files, train_names = load_split_list(dataset_dir, "train.json")
    val_files, val_names = load_split_list(dataset_dir, "val.json")
    label_map = build_label_map(train_names, val_names)
    with open(output_dir / "labels_transformer.json", "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

    y_train = [label_map[n] for n in train_names]
    y_val = [label_map[n] for n in val_names]

    train_ds = TrainChunkDataset(
        train_files, y_train, chunk_sec=args.chunk_sec, augment=bool(args.augment),
        spec_time_mask=args.spec_time_mask, spec_freq_mask=args.spec_freq_mask,
        spec_num_time_masks=args.spec_num_time_masks, spec_num_freq_masks=args.spec_num_freq_masks,
        dataset_root=dataset_dir,
    )
    # Optional class-balanced sampling
    train_sampler = None
    if getattr(args, "balance_sampler", False):
        class_counts: Dict[int, int] = {}
        for yy in y_train:
            class_counts[yy] = class_counts.get(yy, 0) + 1
        weights = [1.0 / class_counts[yy] for yy in y_train]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformer(
        n_mels=N_MELS, n_classes=len(label_map), depth=args.depth, base_channels=args.base_channels,
        d_model=args.d_model, n_heads=args.tx_heads, n_layers=args.tx_layers, ff_dim=args.tx_ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        warm = max(0, int(args.warmup_epochs))
        if warm > 0 and args.epochs > 1:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warm
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warm), eta_min=args.cosine_eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warm]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.cosine_eta_min
            )

    if args.wandb and WANDB_AVAILABLE and run is not None:
        wandb.watch(model, log="gradients", log_freq=100)
        wandb.summary["n_params"] = int(sum(p.numel() for p in model.parameters()))

    # Run directory
    run_id = getattr(run, "id", None) if run is not None else None
    if not run_id:
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{str(uuid4())[:8]}"
    run_dir = output_dir / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0
    eval_overlap = args.overlap if args.eval_overlap is None else float(args.eval_overlap)
    eval_max_segments = None if (getattr(args, "eval_max_segments", 0) or 0) <= 0 else int(args.eval_max_segments)

    for epoch in range(1, args.epochs + 1):
        avg_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
            mixup_alpha=float(getattr(args, "mixup_alpha", 0.0)),
            mixup_p=float(getattr(args, "mixup_p", 1.0)),
            clip_grad_norm=float(getattr(args, "clip_grad_norm", 0.0)),
            amp=bool(args.amp),
        )
        should_eval = (epoch % max(1, int(args.eval_every)) == 0) or (epoch == args.epochs)
        val_logged: Dict[str, float] = {}
        if should_eval:
            top1, top3, val_loss = evaluate_trackwise(
                model, val_files, y_val, device,
                chunk_sec=args.chunk_sec, overlap=eval_overlap,
                batch_size=args.eval_batch_size, max_segments=eval_max_segments,
                agg=str(getattr(args, "eval_agg", "mean")),
            )
            print(f"Epoch {epoch}: loss={avg_loss:.4f} | train@top1={train_acc:.4f} | Val loss={val_loss:.4f} | Val Top-1={top1:.4f}, Top-3={top3:.4f}")
            val_logged = {"val/top1": top1, "val/top3": top3, "val/loss": val_loss}
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f} | train@top1={train_acc:.4f}")

        if args.wandb and WANDB_AVAILABLE and run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/top1": train_acc,
                "lr": current_lr,
            }
            log_dict.update(val_logged)
            wandb.log(log_dict)

        if scheduler is not None:
            scheduler.step()

        if should_eval and val_logged:
            top1 = val_logged["val/top1"]
            top3 = val_logged["val/top3"]
            if top1 > best_top1:
                best_top1 = top1
                ckpt = {
                    "model": model.state_dict(),
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_top1": top1,
                    "val_top3": top3,
                }
                best_path = run_dir / "best_model.pt"
                torch.save(ckpt, best_path)
                print(f"Saved best model to {best_path}")
                if args.wandb and WANDB_AVAILABLE and run is not None:
                    wandb.summary["best_val_top1"] = top1
                    wandb.summary["best_val_top3"] = top3
                    try:
                        wandb.save(str(best_path))
                    except Exception:
                        pass

    # Optional test predictions if present
    test_dir = dataset_dir / "test"
    test_files = []
    if test_dir.exists():
        test_files = sorted(list(test_dir.glob("*.wav")) + list(test_dir.glob("*.mp3")))
    if test_files:
        print(f"Generating predictions for {len(test_files)} test tracks...")
        model.eval()
        pred_dict: Dict[str, List[str]] = {}
        inv_label = {v: k for k, v in label_map.items()}
        for p in tqdm(test_files, desc="Test", ncols=80):
            wav = load_audio(str(p))
            chunks = slide_chunks(wav, chunk_sec=args.chunk_sec, overlap=eval_overlap)
            if eval_max_segments is not None and eval_max_segments > 0 and len(chunks) > eval_max_segments:
                idxs = np.linspace(0, len(chunks) - 1, num=eval_max_segments, dtype=int).tolist()
                chunks = [chunks[i] for i in idxs]
            logps = []
            for i in range(0, len(chunks), args.eval_batch_size):
                batch = chunks[i:i + args.eval_batch_size]
                feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]
                x = torch.stack(feats, dim=0).to(device)
                logits = model(x)
                logp = F.log_softmax(logits, dim=-1).detach().cpu()
                logps.append(logp)
            logprobs = torch.cat(logps, dim=0)
            if str(getattr(args, "eval_agg", "mean")) == "median":
                logprob_mean = torch.median(logprobs, dim=0).values
            else:
                logprob_mean = logprobs.mean(dim=0)
            topk = torch.topk(logprob_mean, k=3).indices.tolist()
            pred_dict[p.stem] = [inv_label[i] for i in topk]
        pred_path = run_dir / "test_pred.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred_dict, f, ensure_ascii=False, indent=2)
        print(f"Wrote predictions to {pred_path}")
        if args.wandb and WANDB_AVAILABLE and run is not None:
            try:
                wandb.save(str(pred_path))
            except Exception:
                pass


if __name__ == "__main__":
    main()
