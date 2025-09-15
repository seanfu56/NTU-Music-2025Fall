#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torchaudio
from tqdm import tqdm


TARGET_SR = 16000


def read_split_list(artist20_dir: Path, json_name: str) -> List[Path]:
    p = artist20_dir / json_name
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    paths: List[Path] = []
    for rel in rel_list:
        fp = artist20_dir / rel.lstrip("./")
        if fp.exists():
            paths.append(fp)
    return paths


def collect_inputs(artist20_dir: Path) -> List[Path]:
    files: List[Path] = []
    files += read_split_list(artist20_dir, "train.json")
    files += read_split_list(artist20_dir, "val.json")
    # test: include all common audio files under test/
    test_dir = artist20_dir / "test"
    if test_dir.exists():
        for ext in ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a"):
            files += list(test_dir.glob(ext))
    # dedup while preserving order
    seen = set()
    uniq: List[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def ensure_jsons(src_artist20: Path, dst_artist20: Path):
    dst_artist20.mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "val.json"):
        src = src_artist20 / name
        if src.exists():
            dst = dst_artist20 / name
            if not dst.exists():
                shutil.copy2(src, dst)


def save_wav(wav: np.ndarray, sr: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to mono float32 tensor at TARGET_SR
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    wav_t = torchaudio.functional.resample(
        torchaudio.functional.tensor_to_audio(wav.astype(np.float32)), sr, TARGET_SR
    )
    # wav_t is 1D; make shape [1, T]
    wav_t = wav_t.unsqueeze(0)
    torchaudio.save(str(out_path), wav_t, TARGET_SR)


def method_hpss(in_paths: List[Path], src_root: Path, out_root: Path):
    import librosa
    import torch

    for ip in tqdm(in_paths, desc="HPSS", ncols=80):
        try:
            y, sr = librosa.load(str(ip), sr=None, mono=True)
            y_h, _ = librosa.effects.hpss(y)
            rel = ip.relative_to(src_root)
            out = out_root / rel
            out = out.with_suffix(".wav")
            out.parent.mkdir(parents=True, exist_ok=True)
            # to torch tensor [1, T]
            y_t = torch.from_numpy(y_h.astype(np.float32)).unsqueeze(0)
            if sr != TARGET_SR:
                y_t = torchaudio.functional.resample(y_t, sr, TARGET_SR)
            torchaudio.save(str(out), y_t, TARGET_SR)
        except Exception as e:
            print(f"[warn] HPSS failed for {ip}: {e}")


def has_demucs_cli() -> bool:
    try:
        r = subprocess.run([sys.executable, "-m", "demucs.separate", "--help"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0
    except Exception:
        return False


def method_demucs(in_paths: List[Path], src_root: Path, out_root: Path, model: str, jobs: int):
    if not has_demucs_cli():
        raise SystemExit("Demucs is not installed. Run: pip install demucs")

    # Run demucs once over all inputs to a temporary directory
    with tempfile.TemporaryDirectory(prefix="demucs_out_") as tmpdir:
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", model,
            "--two-stems", "vocals",
            "-j", str(jobs),
            "-o", tmpdir,
        ] + [str(p) for p in in_paths]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"demucs returned code {proc.returncode}")

        # Move/convert vocals to mirrored tree
        tmp_root = Path(tmpdir) / model
        for ip in tqdm(in_paths, desc="Collect", ncols=80):
            stem = ip.stem
            # demucs writes: tmp_root/<stem>/vocals.wav
            cand = tmp_root / stem / "vocals.wav"
            if not cand.exists():
                # sometimes nested differently, search
                matches = list(tmp_root.glob(f"**/{stem}/vocals.*"))
                if matches:
                    cand = matches[0]
            if not cand.exists():
                print(f"[warn] Missing demucs output for {ip}")
                continue
            rel = ip.relative_to(src_root)
            out = out_root / rel
            out = out.with_suffix(".wav")
            out.parent.mkdir(parents=True, exist_ok=True)

            try:
                wav, sr = torchaudio.load(str(cand))
                if wav.dim() == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != TARGET_SR:
                    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
                torchaudio.save(str(out), wav, TARGET_SR)
            except Exception as e:
                print(f"[warn] Could not save {out}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Preprocess dataset to vocals-only with same structure.")
    ap.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--dataset", type=str, default="artist20", help="Dataset subdir under data_root")
    ap.add_argument("--out_dataset", type=str, default="artist20_vocals", help="Output dataset subdir under data_root")
    ap.add_argument("--method", type=str, choices=["demucs", "hpss"], default="demucs")
    ap.add_argument("--demucs_model", type=str, default="htdemucs_ft")
    ap.add_argument("--jobs", type=int, default=2, help="Workers for Demucs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    src_artist20 = data_root / args.dataset
    dst_artist20 = data_root / args.out_dataset
    if not src_artist20.exists():
        raise SystemExit(f"Source dataset not found: {src_artist20}")

    inputs = collect_inputs(src_artist20)
    if not inputs:
        print("[warn] No inputs found via splits/test; scanning entire tree for audio files...")
        for ext in ("**/*.mp3", "**/*.wav", "**/*.flac", "**/*.ogg", "**/*.m4a"):
            inputs += list(src_artist20.glob(ext))
        inputs = sorted(set(inputs))
    print(f"Found {len(inputs)} files to process.")

    # Mirror JSONs for completeness
    ensure_jsons(src_artist20, dst_artist20)

    if args.method == "demucs":
        method_demucs(inputs, src_artist20, dst_artist20, args.demucs_model, args.jobs)
    else:
        method_hpss(inputs, src_artist20, dst_artist20)

    print(f"Done. Vocals-only dataset at: {dst_artist20}")
    print("Train with:")
    print(f"  python train.py --data_root {data_root} --use_vocals_only --vocals_root {dst_artist20}")


if __name__ == "__main__":
    main()
