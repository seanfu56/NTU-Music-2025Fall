import argparse
import json
import os
import time
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Tuple
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


# =============================
# Utilities: splits and labels
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


def load_split_list(artist20_dir: Path, json_name: str) -> Tuple[List[str], List[str]]:
    with open(artist20_dir / json_name, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    full_paths: List[str] = []
    labels: List[str] = []
    for rel in rel_list:
        p = artist20_dir / rel.lstrip("./")
        if not p.exists():
            print(f"[warn] Missing file listed in {json_name}: {rel}")
            continue
        full_paths.append(str(p))
        labels.append(_artist_from_path(p))
    return full_paths, labels


def build_label_map(train_labels: List[str], val_labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(train_labels + val_labels))
    return {name: idx for idx, name in enumerate(uniq)}


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


def try_vocals_path(orig_path: str, artist_root: Path | None, vocals_root: Path | None) -> str:
    """Return path to vocals-only counterpart if it exists, else original path.
    The mapping mirrors the artist dataset structure under vocals_root. If extension
    differs, we try common audio extensions while keeping the same stem.
    """
    if vocals_root is None or artist_root is None:
        return orig_path
    try:
        rel = os.path.relpath(orig_path, start=str(artist_root))
    except Exception:
        return orig_path
    base = vocals_root / rel
    if base.exists():
        return str(base)
    # Try alternate extensions with same stem
    stem = Path(rel).with_suffix("")
    for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        cand = vocals_root / (str(stem) + ext)
        if cand.exists():
            return str(cand)
    # Fallback: common Demucs layout vocals_root/<model>/<stem>/vocals.ext
    stem_name = Path(orig_path).stem
    pattern = str(vocals_root / "**" / stem_name / "vocals.*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return orig_path


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
    # wav: [T]
    x = mel_spec(wav)  # [n_mels, frames]
    x = amp_to_db(x)
    # per-utterance normalization
    m = x.mean()
    s = x.std()
    x = (x - m) / (s + 1e-5)
    return x  # [n_mels, frames]


def slice_random_chunk(wav: torch.Tensor, chunk_sec: float = 3.0, sr: int = TARGET_SR) -> torch.Tensor:
    chunk_len = int(chunk_sec * sr)
    if wav.numel() < chunk_len:
        # repeat-pad
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


# =============================
# Datasets
# =============================

class TrainChunkDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], chunk_sec: float = 3.0,
                 augment: bool = True, spec_time_mask: int = 30, spec_freq_mask: int = 24,
                 spec_num_time_masks: int = 2, spec_num_freq_masks: int = 2,
                 artist_root: Path | None = None, vocals_root: Path | None = None, prefer_vocals: bool = False):
        self.files = files
        self.labels = labels
        self.chunk_sec = chunk_sec
        self.augment = augment
        self.spec_time_mask = int(spec_time_mask)
        self.spec_freq_mask = int(spec_freq_mask)
        self.spec_num_time_masks = int(spec_num_time_masks)
        self.spec_num_freq_masks = int(spec_num_freq_masks)
        self.artist_root = artist_root
        self.vocals_root = vocals_root
        self.prefer_vocals = prefer_vocals

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        if self.prefer_vocals:
            path = try_vocals_path(path, self.artist_root, self.vocals_root)
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
# Model: Short-chunk CNN
# =============================

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # pool over frequency only to preserve time resolution
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        # projection for residual path when shape changes
        self.proj = None
        self.use_proj = in_ch != out_ch
        if self.use_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_ch)
        else:
            self.proj_bn = None

    def forward(self, x):  # x: [B, C, M, T]
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        # Only pool over frequency when it is >= 2 to avoid collapsing to 0
        do_pool = out.size(-2) >= 2
        if do_pool:
            out = self.pool(out)
        # residual path: match channels and pooled freq dim
        if self.use_proj and self.proj is not None:
            identity = self.proj(identity)
            identity = self.proj_bn(identity)  # type: ignore[arg-type]
        if do_pool:
            identity = self.pool(identity)
        out = out + identity
        out = self.act(out)
        return out


class ShortChunkCNN(nn.Module):
    def __init__(self, n_mels: int, n_classes: int, *, depth: int = 4, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        depth = max(1, int(depth))
        base_channels = max(8, int(base_channels))
        blocks = []
        in_c = 1
        out_c = base_channels
        for _ in range(depth):
            blocks.append(ResidualConvBlock(in_c, out_c))
            in_c = out_c
            out_c = out_c * 2
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.last_channels = in_c  # channels after final block
        hidden = max(128, self.last_channels)
        self.head = nn.Sequential(
            nn.Linear(self.last_channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):  # x: [B, 1, M, T]
        for blk in self.blocks:
            x = blk(x)
        # x: [B, C, M', T]
        mean_t = x.mean(dim=-1)
        std_t = x.std(dim=-1)
        mean_t = mean_t.mean(dim=-1)
        std_t = std_t.mean(dim=-1)
        feat = torch.cat([mean_t, std_t], dim=1)
        feat = self.dropout(feat)
        logits = self.head(feat)
        return logits


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
    batch_size: int = 16,
    max_segments: int | None = None,
    artist_root: Path | None = None,
    vocals_root: Path | None = None,
    prefer_vocals: bool = False,
) -> Tuple[float, float, float]:
    model.eval()
    all_top1 = []
    all_top3 = []
    val_losses = []
    for path, y in tqdm(list(zip(files, labels)), desc="Eval", ncols=80):
        if prefer_vocals:
            path = try_vocals_path(path, artist_root, vocals_root)
        wav = load_audio(path)
        chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
        # Optionally cap the number of segments for faster evaluation
        if max_segments is not None and max_segments > 0 and len(chunks) > max_segments:
            # Uniformly sample indices across the full track to preserve coverage
            idxs = np.linspace(0, len(chunks) - 1, num=max_segments, dtype=int).tolist()
            chunks = [chunks[i] for i in idxs]
        # batch inference over chunks for speed
        logprob_list = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]  # [1,M,T]
            x = torch.stack(feats, dim=0).to(device)  # [B,1,M,T]
            logits = model(x)
            logp = F.log_softmax(logits, dim=-1).detach().cpu()
            logprob_list.append(logp)
        logprobs = torch.cat(logprob_list, dim=0)  # [Nseg, C]
        logprob_mean = logprobs.mean(dim=0)  # [C]
        top3 = torch.topk(logprob_mean, k=3).indices.tolist()
        pred = top3[0]
        all_top1.append(1 if pred == y else 0)
        all_top3.append(1 if y in top3 else 0)
        val_losses.append(float(-logprob_mean[int(y)]))
    top1 = float(np.mean(all_top1)) if all_top1 else 0.0
    top3 = float(np.mean(all_top3)) if all_top3 else 0.0
    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    return top1, top3, val_loss


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total = 0.0
    n = 0
    correct = 0
    for feats, y in tqdm(loader, desc="Train", ncols=80):
        feats = feats.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(feats)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == y).sum().item()
    avg_loss = total / max(1, n)
    acc = correct / max(1, n)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Short-chunk CNN singer classifier (PyTorch)")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5, help="Eval window overlap (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    # Vocal preprocessing
    parser.add_argument("--vocals_root", type=str, default=None,
                        help="Root dir mirroring artist20 with vocals-only audio")
    parser.add_argument("--use_vocals_only", action="store_true",
                        help="Prefer vocals-only files if available (fallback to mix)")
    # Model capacity
    parser.add_argument("--depth", type=int, default=4, help="Number of Conv blocks")
    parser.add_argument("--base_channels", type=int, default=32, help="Channels of first Conv block")
    # Optimizer
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # Augmentation
    parser.add_argument("--augment", action="store_true", default=True, help="Enable SpecAugment during training")
    parser.add_argument("--no_augment", action="store_false", dest="augment", help="Disable SpecAugment")
    parser.add_argument("--spec_time_mask", type=int, default=30)
    parser.add_argument("--spec_freq_mask", type=int, default=24)
    parser.add_argument("--spec_num_time_masks", type=int, default=2)
    parser.add_argument("--spec_num_freq_masks", type=int, default=2)
    # Evaluation controls
    parser.add_argument("--eval_overlap", type=float, default=None,
                        help="Override overlap for validation/test. If unset, use --overlap.")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Batch size for evaluation forward passes")
    parser.add_argument("--eval_max_segments", type=int, default=64,
                        help="Cap segments per track during eval (0 disables cap)")
    # LR scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"],
                        help="Learning rate scheduler type")
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6,
                        help="eta_min for CosineAnnealingLR")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs before cosine")
    # Evaluation frequency
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Run validation every N epochs (always runs at last epoch)")
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="ntu-music-singer-cnn")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run = None
    if args.wandb and WANDB_AVAILABLE:
        # Detect if launched by a sweep agent: don't pass a config to init; let sweep YAML drive it
        is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))
        if is_sweep:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
            )
        else:
            # Single run with wandb: log the actual scalar args as config
            base_config = {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "depth": args.depth,
                "base_channels": args.base_channels,
                "lr_scheduler": args.lr_scheduler,
                "cosine_eta_min": args.cosine_eta_min,
                "warmup_epochs": args.warmup_epochs,
                "num_workers": args.num_workers,
                "chunk_sec": args.chunk_sec,
                "overlap": args.overlap,
                "eval_every": args.eval_every,
                "augment": args.augment,
                "spec_time_mask": args.spec_time_mask,
                "spec_freq_mask": args.spec_freq_mask,
                "spec_num_time_masks": args.spec_num_time_masks,
                "spec_num_freq_masks": args.spec_num_freq_masks,
                "seed": args.seed,
            }
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=base_config,
            )
        # Override args from sweep config if present (sweeps inject scalars)
        cfg = wandb.config
        args.batch_size = int(cfg.get("batch_size", args.batch_size))
        args.epochs = int(cfg.get("epochs", args.epochs))
        args.lr = float(cfg.get("lr", args.lr))
        args.num_workers = int(cfg.get("num_workers", args.num_workers))
        args.chunk_sec = float(cfg.get("chunk_sec", args.chunk_sec))
        args.overlap = float(cfg.get("overlap", args.overlap))
        args.seed = int(cfg.get("seed", args.seed))
        # Vocals preprocessing overrides
        if "vocals_root" in cfg and cfg["vocals_root"]:
            args.vocals_root = str(cfg["vocals_root"])  # type: ignore[attr-defined]
        if "use_vocals_only" in cfg:
            args.use_vocals_only = bool(cfg.get("use_vocals_only", args.use_vocals_only))
        # Scheduler overrides
        args.lr_scheduler = str(cfg.get("lr_scheduler", args.lr_scheduler))
        args.cosine_eta_min = float(cfg.get("cosine_eta_min", args.cosine_eta_min))
        args.warmup_epochs = int(cfg.get("warmup_epochs", args.warmup_epochs))
        # Model capacity overrides
        args.depth = int(cfg.get("depth", args.depth))
        args.base_channels = int(cfg.get("base_channels", args.base_channels))
        # Optimizer
        args.weight_decay = float(cfg.get("weight_decay", args.weight_decay))
        # Augmentation
        if "augment" in cfg:
            args.augment = bool(cfg.get("augment", args.augment))
        args.spec_time_mask = int(cfg.get("spec_time_mask", args.spec_time_mask))
        args.spec_freq_mask = int(cfg.get("spec_freq_mask", args.spec_freq_mask))
        args.spec_num_time_masks = int(cfg.get("spec_num_time_masks", args.spec_num_time_masks))
        args.spec_num_freq_masks = int(cfg.get("spec_num_freq_masks", args.spec_num_freq_masks))
        # Optional eval-specific overrides from sweep
        if "eval_overlap" in cfg and cfg["eval_overlap"] is not None:
            args.eval_overlap = float(cfg["eval_overlap"])  # type: ignore[attr-defined]
        if "eval_batch_size" in cfg and cfg["eval_batch_size"] is not None:
            args.eval_batch_size = int(cfg["eval_batch_size"])  # type: ignore[attr-defined]
        if "eval_max_segments" in cfg and cfg["eval_max_segments"] is not None:
            args.eval_max_segments = int(cfg["eval_max_segments"])  # type: ignore[attr-defined]

    data_root = Path(args.data_root)
    artist20 = data_root / "artist20_vocals"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, train_names = load_split_list(artist20, "train.json")
    val_files, val_names = load_split_list(artist20, "val.json")
    label_map = build_label_map(train_names, val_names)
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

    y_train = [label_map[n] for n in train_names]
    y_val = [label_map[n] for n in val_names]

    train_ds = TrainChunkDataset(
        train_files,
        y_train,
        chunk_sec=args.chunk_sec,
        augment=bool(args.augment),
        spec_time_mask=args.spec_time_mask,
        spec_freq_mask=args.spec_freq_mask,
        spec_num_time_masks=args.spec_num_time_masks,
        spec_num_freq_masks=args.spec_num_freq_masks,
        artist_root=artist20,
        vocals_root=(Path(args.vocals_root) if args.vocals_root else None),
        prefer_vocals=bool(args.use_vocals_only),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortChunkCNN(n_mels=N_MELS, n_classes=len(label_map), depth=args.depth, base_channels=args.base_channels).to(device)
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
        # Log static artifacts/metadata
        n_params = sum(p.numel() for p in model.parameters())
        wandb.summary["n_params"] = int(n_params)

    # Create unique run directory for checkpoints/artifacts
    run_id = None
    if args.wandb and WANDB_AVAILABLE and run is not None:
        run_id = getattr(run, "id", None) or getattr(run, "name", None)
    if not run_id:
        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{str(uuid4())[:8]}"
    run_dir = output_dir / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_top1 = 0.0
    eval_overlap = args.overlap if args.eval_overlap is None else float(args.eval_overlap)
    eval_max_segments = None if (getattr(args, "eval_max_segments", 0) or 0) <= 0 else int(args.eval_max_segments)
    for epoch in range(1, args.epochs + 1):
        avg_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        should_eval = (epoch % max(1, int(args.eval_every)) == 0) or (epoch == args.epochs)
        val_logged = {}
        if should_eval:
            top1, top3, val_loss = evaluate_trackwise(
                model,
                val_files,
                y_val,
                device,
                chunk_sec=args.chunk_sec,
                overlap=eval_overlap,
                batch_size=args.eval_batch_size,
                max_segments=eval_max_segments,
                artist_root=artist20,
                vocals_root=(Path(args.vocals_root) if args.vocals_root else None),
                prefer_vocals=bool(args.use_vocals_only),
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

        # Step LR scheduler per epoch (after validation)
        if scheduler is not None:
            scheduler.step()

        if should_eval:
            if top1 > best_top1:  # type: ignore[name-defined]
                best_top1 = top1  # type: ignore[name-defined]
                ckpt = {
                    "model": model.state_dict(),
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_top1": top1,  # type: ignore[name-defined]
                    "val_top3": top3,  # type: ignore[name-defined]
                }
                best_path = run_dir / "best_model.pt"
                torch.save(ckpt, best_path)
                print(f"Saved best model to {best_path}")
                if args.wandb and WANDB_AVAILABLE and run is not None:
                    wandb.summary["best_val_top1"] = top1  # type: ignore[name-defined]
                    wandb.summary["best_val_top3"] = top3  # type: ignore[name-defined]
                    try:
                        wandb.save(str(best_path))
                    except Exception:
                        pass

    # Optional: generate test predictions (top-3) to match the HW format
    test_dir = artist20 / "test"
    # test_files = sorted([p for p in test_dir.glob("*.mp3")])
    test_files = sorted([p for p in test_dir.glob("*.wav")])

    if test_files:
        print(f"Generating predictions for {len(test_files)} test tracks...")
        model.eval()
        pred_dict: Dict[str, List[str]] = {}
        inv_label = {v: k for k, v in label_map.items()}
        for p in tqdm(test_files, desc="Test", ncols=80):
            test_path = str(p)
            if bool(args.use_vocals_only):
                test_path = try_vocals_path(test_path, artist20, (Path(args.vocals_root) if args.vocals_root else None))
            wav = load_audio(test_path)
            chunks = slide_chunks(wav, chunk_sec=args.chunk_sec, overlap=eval_overlap)
            if eval_max_segments is not None and eval_max_segments > 0 and len(chunks) > eval_max_segments:
                idxs = np.linspace(0, len(chunks) - 1, num=eval_max_segments, dtype=int).tolist()
                chunks = [chunks[i] for i in idxs]
            probs_list = []
            for i in range(0, len(chunks), args.eval_batch_size):
                batch = chunks[i:i + args.eval_batch_size]
                feats = [wav_to_logmel(ch).unsqueeze(0) for ch in batch]
                x = torch.stack(feats, dim=0).to(device)
                logits = model(x)
                logp = F.log_softmax(logits, dim=-1).detach().cpu()
                probs_list.append(logp)
            logprobs = torch.cat(probs_list, dim=0)
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
