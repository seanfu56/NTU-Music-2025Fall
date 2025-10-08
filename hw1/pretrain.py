#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================
# 全域設定
# =============================
TARGET_SR = 16000  # wav2vec2 預設訓練是 16kHz
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 資料處理：讀檔、切片、遞迴抓檔
# =============================
def load_audio_mono16k(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [T]

def slice_random_chunk(wav: torch.Tensor, chunk_sec: float = 4.0, sr: int = TARGET_SR) -> torch.Tensor:
    L = int(chunk_sec * sr)
    if wav.numel() < L:
        reps = (L + wav.numel() - 1) // wav.numel()
        wav = wav.repeat(reps)[:L]
        return wav
    start = int(torch.randint(0, wav.numel() - L + 1, (1,)).item())
    return wav[start:start+L]

def slide_chunks(wav: torch.Tensor, chunk_sec: float = 4.0, overlap: float = 0.5, sr: int = TARGET_SR) -> List[torch.Tensor]:
    L = int(chunk_sec * sr)
    if wav.numel() <= L:
        return [slice_random_chunk(wav, chunk_sec, sr)]
    step = max(1, int(L * (1 - overlap)))
    starts = list(range(0, max(1, wav.numel() - L + 1), step))
    if starts[-1] != wav.numel() - L:
        starts.append(wav.numel() - L)
    return [wav[s:s+L] for s in starts]

def list_audio_files_recursive(root: Path, exts=(".mp3", ".wav", ".flac", ".m4a", ".ogg")) -> List[Path]:
    exts = tuple(e.lower() for e in exts)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: str(p).lower())
    return files

# =============================
# Dataset / Collate
# =============================
class WaveChunkDataset(Dataset):
    """回傳 raw waveform chunk（不做梅爾/頻譜）。"""
    def __init__(self, files: List[str], labels: List[int], chunk_sec: float = 4.0):
        self.files = files
        self.labels = labels
        self.chunk_sec = float(chunk_sec)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        y = self.labels[idx]
        wav = load_audio_mono16k(self.files[idx])
        chunk = slice_random_chunk(wav, self.chunk_sec)
        return chunk, y  # [T], int

def pad_collate(batch, pad_value: float = 0.0):
    # batch: List[(wav[T], y)]
    ys = []
    lens = []
    T_max = 0
    for wav, y in batch:
        ys.append(int(y))
        lens.append(len(wav))
        T_max = max(T_max, len(wav))
    padded = torch.full((len(batch), T_max), pad_value, dtype=torch.float32)
    for i, (wav, _) in enumerate(batch):
        padded[i, :len(wav)] = wav
    lengths = torch.tensor(lens, dtype=torch.int32)
    ys = torch.tensor(ys, dtype=torch.long)
    return padded, lengths, ys  # [B,T], [B], [B]

# =============================
# Label Utils（承接你原本的 split 檔）
# =============================
def _artist_from_path(p: Path) -> str:
    # 自動從路徑抓類別名（可依你的資料結構自行調整）
    # e.g. .../train_val/<artist>/<song>.mp3
    parts = list(p.parts)
    if "train_val" in parts:
        try:
            idx = parts.index("train_val")
            return parts[idx + 1]
        except Exception:
            pass
    return p.parent.parent.name

def load_split_list(dataset_dir: Path, json_name: str, fallback_exts=(".mp3", ".wav")) -> Tuple[List[str], List[str]]:
    with open(dataset_dir / json_name, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    files, labels = [], []
    for rel in rel_list:
        base = (dataset_dir / rel.lstrip("./"))
        # 嘗試多種副檔名（你資料以前用 .wav，若改成 mp3，仍可找到）
        found = None
        for ext in fallback_exts:
            p = base.with_suffix(ext)
            if p.exists():
                found = p
                break
        if found is None:
            # 如果 split 檔提供的是完整檔名含副檔名，就直接用
            if base.exists():
                found = base
            else:
                print(f"[warn] Missing file listed in {json_name}: {rel}")
                continue
        files.append(str(found))
        labels.append(_artist_from_path(found))
    return files, labels

def build_label_map(train_labels: List[str], val_labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(train_labels + val_labels))
    return {name: idx for idx, name in enumerate(uniq)}

def gather_inference_targets(
    dataset_dir: Path,
    manifest_path: Optional[Path],
    infer_dir: Optional[Path],
    fallback_exts: Sequence[str] = (".mp3", ".wav", ".flac", ".m4a", ".ogg"),
) -> List[Tuple[str, Path]]:
    """
    Collect audio files for inference.
    Returns list of tuples (relative_key, absolute_path).
    """
    targets: List[Tuple[str, Path]] = []
    seen_real: set[str] = set()

    if manifest_path is not None:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"Inference manifest must be a list of paths, got {type(entries)}")
        for rel in entries:
            if not isinstance(rel, str):
                print(f"[warn] Skip non-string entry in manifest: {rel}")
                continue
            rel_clean = rel.lstrip("./")
            base = dataset_dir / rel_clean
            found: Optional[Path] = None
            if base.exists():
                found = base
            else:
                for ext in fallback_exts:
                    cand = base.with_suffix(ext)
                    if cand.exists():
                        found = cand
                        break
            if found is None:
                print(f"[warn] Manifest item not found on disk: {rel}")
                continue
            real_path = str(found.resolve())
            if real_path in seen_real:
                continue
            rel_key = found.relative_to(dataset_dir).as_posix()
            targets.append((rel_key, found))
            seen_real.add(real_path)

    if infer_dir is not None:
        root = infer_dir.resolve()
        for p in list_audio_files_recursive(root):
            real_path = str(p.resolve())
            if real_path in seen_real:
                continue
            rel_key = p.relative_to(root).as_posix()
            targets.append((rel_key, p))
            seen_real.add(real_path)

    return targets

# =============================
# wav2vec2 Backbone（torchaudio 或 HF 2 選 1）
# =============================
class Wav2Vec2Backbone(nn.Module):
    """
    以 torchaudio 的預訓練 wav2vec 2.0 為主，選項：
    - WAV2VEC2_BASE
    - WAV2VEC2_LARGE
    - WAV2VEC2_XLSR53
    若想用 Hugging Face，請帶 --backbone hf 並安裝 transformers。
    """
    def __init__(self, source: str = "torchaudio", bundle_name: str = "WAV2VEC2_BASE", hf_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.source = source
        self.hidden_size: Optional[int] = None

        if source == "torchaudio":
            bundles = {
                "WAV2VEC2_BASE": torchaudio.pipelines.WAV2VEC2_BASE,
                "WAV2VEC2_LARGE": torchaudio.pipelines.WAV2VEC2_LARGE,
                "WAV2VEC2_XLSR53": torchaudio.pipelines.WAV2VEC2_XLSR53,
            }
            if bundle_name not in bundles:
                raise ValueError(f"Unknown torchaudio bundle: {bundle_name}")
            self.bundle = bundles[bundle_name]
            self.model = self.bundle.get_model()  # Wav2Vec2Model

            # ★不要用固定屬性名稱抓 hidden dim；直接用假輸入跑一次 extract_features
            with torch.no_grad():
                dummy = torch.zeros(1, 16000, dtype=torch.float32)  # 1 秒 16kHz
                feats, out_lens = self.model.extract_features(dummy)  # List[Tensor], lengths
                last = feats[-1]  # [B, S, C]
                self.hidden_size = int(last.shape[-1])
        else:
            # Hugging Face
            try:
                from transformers import Wav2Vec2Model
            except Exception as e:
                raise RuntimeError("transformers 尚未安裝，請先 pip install transformers") from e
            self.hf = True
            self.model = Wav2Vec2Model.from_pretrained(hf_name)
            self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward_features(self, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        輸入:
            wav: [B, T]（float32, 16k mono）
            lengths: [B] 實際長度
        輸出:
            x: [B, S, C]（時間步 S 為下采樣後的長度；C=hidden_size）
        """
        self.model.eval()
        if self.source == "torchaudio":
            feats, out_lens = self.model.extract_features(wav, lengths)
            x = feats[-1]  # 取最後一層 encoder 輸出 [B, S, C]
            return x
        else:
            attn_mask = None
            if lengths is not None:
                max_len = wav.size(1)
                mask = torch.arange(max_len, device=wav.device)[None, :] < lengths[:, None]
                attn_mask = mask.long()
            out = self.model(wav, attention_mask=attn_mask)
            x = out.last_hidden_state  # [B, S, C]
            return x

# =============================
# 分類模型：wav2vec2 + 池化 + MLP
# =============================
class W2V2Classifier(nn.Module):
    def __init__(self, backbone: Wav2Vec2Backbone, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        hs = int(backbone.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hs * 2, hs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hs, num_classes),
        )

    def forward(self, wav: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x = self.backbone.forward_features(wav, lengths)  # [B,S,C]
        mean_t = x.mean(dim=1)
        std_t = x.std(dim=1)
        feat = torch.cat([mean_t, std_t], dim=1)  # [B,2C]
        logits = self.head(feat)
        return logits

# =============================
# 訓練 / 驗證
# =============================
@torch.no_grad()
def evaluate_trackwise(
    model: nn.Module,
    files: List[str],
    labels: List[int],
    chunk_sec: float = 4.0,
    overlap: float = 0.5,
    batch_size: int = 32,
) -> Tuple[float, float, float]:
    model.eval()
    all_top1, all_top3, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for path, y in tqdm(list(zip(files, labels)), desc="Eval", ncols=80):
        wav = load_audio_mono16k(path)
        chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
        logps = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            padded, lens, _ = pad_collate([(b, y) for b in batch])
            padded = padded.to(TORCH_DEVICE)
            lens = lens.to(TORCH_DEVICE)
            logits = model(padded, lens)  # [B,C]
            logp = F.log_softmax(logits, dim=-1).detach().cpu()
            logps.append(logp)
        lp = torch.cat(logps, dim=0).mean(dim=0)  # mean 聚合
        pred_top3 = torch.topk(lp, k=min(3, lp.numel())).indices.tolist()
        losses.append(float(-lp[int(y)]))
        all_top1.append(1 if pred_top3[0] == y else 0)
        all_top3.append(1 if y in pred_top3 else 0)
    return float(np.mean(all_top1)), float(np.mean(all_top3)), float(np.mean(losses))

@torch.no_grad()
def predict_trackwise(
    model: nn.Module,
    targets: List[Tuple[str, Path]],
    inv_label_map: Dict[int, str],
    chunk_sec: float = 4.0,
    overlap: float = 0.5,
    batch_size: int = 32,
    topk: int = 3,
) -> Dict[str, List[str]]:
    """
    Produce top-k label predictions for each audio track.
    """
    model.eval()
    results: Dict[str, List[str]] = {}
    for key, audio_path in tqdm(targets, desc="Infer", ncols=80):
        try:
            wav = load_audio_mono16k(str(audio_path))
        except Exception as e:
            print(f"[warn] Failed to load {audio_path}: {e}")
            continue
        chunks = slide_chunks(wav, chunk_sec=chunk_sec, overlap=overlap)
        if not chunks:
            print(f"[warn] No chunks generated for {audio_path}")
            continue
        logps = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            padded, lens, _ = pad_collate([(b, 0) for b in batch])
            padded = padded.to(TORCH_DEVICE)
            lens = lens.to(TORCH_DEVICE)
            logits = model(padded, lens)
            logp = F.log_softmax(logits, dim=-1).detach().cpu()
            logps.append(logp)
        if not logps:
            continue
        lp = torch.cat(logps, dim=0).mean(dim=0)
        kk = min(topk, lp.numel())
        top_indices = torch.topk(lp, k=kk).indices.tolist()
        results[key] = [inv_label_map[int(idx)] for idx in top_indices]
    return results

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    amp: bool = True,
    clip_grad_norm: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and TORCH_DEVICE.type == "cuda")
    total, n, correct = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    for wav, lengths, y in tqdm(loader, desc="Train", ncols=80):
        wav = wav.to(TORCH_DEVICE)
        lengths = lengths.to(TORCH_DEVICE)
        y = y.to(TORCH_DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and TORCH_DEVICE.type == "cuda"):
            logits = model(wav, lengths)  # [B,C]
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * y.size(0)
        n += y.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == y).sum().item()
    return total / max(1, n), correct / max(1, n)

# =============================
# 主流程
# =============================
def main():
    ap = argparse.ArgumentParser("Singer classification with wav2vec 2.0 (pretrained backbone)")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--dataset_subdir", type=str, default="artist20_vocals")
    ap.add_argument("--output_dir", type=str, default="output_w2v2")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"],
                    help="train: fit the model; infer: run inference using a checkpoint")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4, help="head lr（若未設 head_lr/backbone_lr 則兩者皆用此）")
    ap.add_argument("--head_lr", type=float, default=None)
    ap.add_argument("--backbone_lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--chunk_sec", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    # wav2vec 選擇
    ap.add_argument("--backbone", type=str, default="torchaudio", choices=["torchaudio", "hf"])
    ap.add_argument("--bundle", type=str, default="WAV2VEC2_BASE",
                    help="torchaudio: WAV2VEC2_BASE | WAV2VEC2_LARGE | WAV2VEC2_XLSR53")
    ap.add_argument("--hf_name", type=str, default="facebook/wav2vec2-base-960h",
                    help="transformers model name if --backbone=hf")
    # 凍結/解凍
    ap.add_argument("--freeze_backbone", action="store_true", help="先凍結 wav2vec2 backbone，只訓練 head")
    ap.add_argument("--unfreeze_at", type=int, default=0, help="第幾個 epoch 開始解凍 backbone（0=不解凍）")
    ap.add_argument("--clip_grad_norm", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", default=True)
    # inference
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (.pt) for inference (default: output_dir/best_w2v2.pt)")
    ap.add_argument("--infer_manifest", type=str, default=None,
                    help="JSON list of relative paths (e.g., val.json) to run inference on")
    ap.add_argument("--infer_dir", type=str, default=None,
                    help="Directory to recursively search for audio files for inference")
    ap.add_argument("--infer_output", type=str, default="predictions.json",
                    help="Output JSON file name (relative to output_dir unless absolute)")
    ap.add_argument("--infer_topk", type=int, default=3, help="Number of labels to report per track")
    args = ap.parse_args()

    if args.head_lr is None:
        args.head_lr = args.lr
    if args.backbone_lr is None:
        args.backbone_lr = args.lr

    data_root = Path(args.data_root)
    dataset_dir = data_root / args.dataset_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if args.mode == "infer":
        ckpt_path = Path(args.checkpoint).expanduser() if args.checkpoint else (output_dir / "best_w2v2.pt")
        ckpt_path = ckpt_path.expanduser()
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

        manifest_path: Optional[Path] = None
        if args.infer_manifest:
            cand = Path(args.infer_manifest).expanduser()
            if not cand.exists():
                cand_rel = (dataset_dir / args.infer_manifest).expanduser()
                if cand_rel.exists():
                    cand = cand_rel
            if not cand.exists():
                raise SystemExit(f"Inference manifest not found: {cand}")
            manifest_path = cand
        else:
            default_manifest = dataset_dir / "val.json"
            if default_manifest.exists():
                manifest_path = default_manifest

        infer_dir: Optional[Path] = None
        if args.infer_dir:
            infer_dir = Path(args.infer_dir).expanduser()
            if not infer_dir.exists():
                raise SystemExit(f"Inference directory not found: {infer_dir}")
        targets = gather_inference_targets(dataset_dir, manifest_path, infer_dir)
        if not targets:
            raise SystemExit("No inference targets found. Provide --infer_manifest or --infer_dir.")

        print(f"[info] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        label_map: Dict[str, int] = ckpt["label_map"]
        inv_label_map = {idx: name for name, idx in label_map.items()}
        backbone_source = ckpt.get("backbone", args.backbone)
        bundle_name = ckpt.get("bundle", args.bundle)
        hf_name = ckpt.get("hf_name", args.hf_name)

        backbone_kwargs = {"source": backbone_source}
        if backbone_source == "torchaudio":
            backbone_kwargs["bundle_name"] = bundle_name or args.bundle
        else:
            backbone_kwargs["hf_name"] = hf_name or args.hf_name
        backbone = Wav2Vec2Backbone(**backbone_kwargs)
        backbone.model.to(TORCH_DEVICE)
        model = W2V2Classifier(backbone, num_classes=len(label_map), dropout=0.2).to(TORCH_DEVICE)
        model.load_state_dict(ckpt["model"])

        predictions = predict_trackwise(
            model,
            targets,
            inv_label_map,
            chunk_sec=float(args.chunk_sec),
            overlap=float(args.overlap),
            batch_size=int(args.eval_batch_size),
            topk=int(args.infer_topk),
        )

        pred_out = Path(args.infer_output).expanduser()
        if not pred_out.is_absolute():
            pred_out = output_dir / pred_out
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_out, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"[info] Wrote predictions to {pred_out}")
        return

    # 讀取 split
    train_files, train_names = load_split_list(dataset_dir, "train.json")
    val_files, val_names = load_split_list(dataset_dir, "val.json")
    label_map = build_label_map(train_names, val_names)
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    y_train = [label_map[n] for n in train_names]
    y_val = [label_map[n] for n in val_names]

    # Dataset / Loader
    train_ds = WaveChunkDataset(train_files, y_train, chunk_sec=args.chunk_sec)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=pad_collate, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Backbone + Classifier
    backbone = Wav2Vec2Backbone(
        source=args.backbone,
        bundle_name=args.bundle,
        hf_name=args.hf_name
    )
    backbone.model.to(TORCH_DEVICE)
    model = W2V2Classifier(backbone, num_classes=len(label_map), dropout=0.2).to(TORCH_DEVICE)

    # 凍結 / 解凍
    def set_backbone_requires_grad(flag: bool):
        for p in model.backbone.model.parameters():
            p.requires_grad = flag

    if args.freeze_backbone:
        set_backbone_requires_grad(False)

    # 分組學習率
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head"):
            head_params.append(p)
        else:
            bb_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": float(args.backbone_lr), "weight_decay": args.weight_decay},
            {"params": head_params, "lr": float(args.head_lr), "weight_decay": args.weight_decay},
        ]
    )

    # 訓練
    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # 漸進解凍
        if args.unfreeze_at > 0 and epoch == int(args.unfreeze_at):
            set_backbone_requires_grad(True)
            # 重新設定 optimizer（確保 requires_grad 生效&學習率分組一致）
            head_params, bb_params = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("head"):
                    head_params.append(p)
                else:
                    bb_params.append(p)
            optim = torch.optim.AdamW(
                [
                    {"params": bb_params, "lr": float(args.backbone_lr), "weight_decay": args.weight_decay},
                    {"params": head_params, "lr": float(args.head_lr), "weight_decay": args.weight_decay},
                ]
            )
            print(f"[info] Unfroze backbone at epoch {epoch}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optim, amp=bool(args.amp),
            clip_grad_norm=float(args.clip_grad_norm),
        )

        do_eval = (epoch % max(1, int(args.eval_every)) == 0) or (epoch == args.epochs)
        if do_eval:
            top1, top3, vloss = evaluate_trackwise(
                model, val_files, y_val,
                chunk_sec=args.chunk_sec, overlap=args.overlap,
                batch_size=args.eval_batch_size
            )
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={vloss:.4f} val@1={top1:.4f} val@3={top3:.4f}")
            if top1 > best_top1:
                best_top1 = top1
                ckpt = {
                    "model": model.state_dict(),
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_top1": top1,
                    "val_top3": top3,
                    "backbone": args.backbone,
                    "bundle": args.bundle if args.backbone == "torchaudio" else None,
                    "hf_name": args.hf_name if args.backbone == "hf" else None,
                }
                save_path = output_dir / "best_w2v2.pt"
                torch.save(ckpt, save_path)
                print(f"[info] Saved best to {save_path}")
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

if __name__ == "__main__":
    main()
