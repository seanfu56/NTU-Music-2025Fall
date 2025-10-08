#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import torchaudio
import torch
from tqdm import tqdm

TARGET_SR = 16000
DEMUCS_MODEL = "htdemucs_ft"


def has_demucs_cli() -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "demucs.separate", "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def load_val_file_list(json_path: Path) -> Tuple[List[Path], List[Path]]:
    with open(json_path, "r", encoding="utf-8") as f:
        rel_list = json.load(f)

    base_dir = json_path.parent
    mp3s: List[Path] = []
    missing: List[Path] = []
    for rel in rel_list:
        rel_clean = rel.lstrip("./")
        cand = (base_dir / rel_clean).resolve()
        if cand.exists():
            mp3s.append(cand)
        else:
            missing.append(cand)
    return mp3s, missing


def demucs_vocals_only(in_paths: List[Path], src_root: Path, out_root: Path,
                       device: str, jobs: int, segment: float):
    if not has_demucs_cli():
        raise SystemExit("Demucs 未安裝，請先：pip install demucs")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(prefix="demucs_out_") as tmpdir:
        tmpdir = Path(tmpdir)
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", DEMUCS_MODEL,
            "--two-stems", "vocals",
            "-d", device,
            "-j", str(jobs),
            "-o", str(tmpdir),
            "--segment", str(int(segment)),
        ] + [str(p) for p in in_paths]

        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"demucs 執行失敗 (exit code {proc.returncode})")

        tmp_root = tmpdir / DEMUCS_MODEL

        for ip in tqdm(in_paths, desc="Collect", ncols=80):
            stem = ip.stem
            cand = tmp_root / stem / "vocals.wav"
            if not cand.exists():
                matches = list(tmp_root.glob(f"**/{stem}/vocals.*"))
                if matches:
                    cand = matches[0]
            if not cand.exists():
                print(f"[warn] 找不到 Demucs 輸出：{ip}")
                continue

            try:
                wav, sr = torchaudio.load(str(cand))
                if wav.dim() == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != TARGET_SR:
                    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

                rel = ip.relative_to(src_root)
                out_path = (out_root / rel).with_suffix(".wav")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(out_path), wav, TARGET_SR)
            except Exception as e:
                print(f"[warn] 無法寫出 {ip}: {e}")


def main():
    default_json = Path(__file__).resolve().parent / "data/artist20/val.json"
    ap = argparse.ArgumentParser(
        description="針對 val.json 裡列出的 .mp3，輸出人聲至 out_dir（保留目錄結構，可用 GPU）"
    )
    ap.add_argument("--json_path", type=str, default=str(default_json),
                    help="val.json 路徑（相對或絕對）")
    ap.add_argument("--out_dir", required=True, type=str, help="輸出資料夾")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                    help="推論裝置（預設 auto：有 GPU 則用 cuda）")
    ap.add_argument("--jobs", type=int, default=2, help="CPU 並行數（解碼/IO），GPU 不會因 jobs 提升")
    ap.add_argument("--segment", type=float, default=7.8,
                    help="分段秒數，越小越省 VRAM（10~20 是常見範圍）")
    args = ap.parse_args()

    json_path = Path(args.json_path).resolve()
    if not json_path.exists():
        raise SystemExit(f"val.json 不存在：{json_path}")

    src_root = json_path.parent.resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mp3s, missing = load_val_file_list(json_path)
    if missing:
        for miss in missing:
            print(f"[warn] val.json 指定的檔案不存在：{miss}")
    if not mp3s:
        print("[warn] val.json 沒有任何可用的檔案")
        return

    print(f"共找到 {len(mp3s)} 個檔案")
    tic = os.times()
    demucs_vocals_only(mp3s, src_root, out_root, args.device, args.jobs, args.segment)
    toc = os.times()
    print(f"完成，結果保存在：{out_root}")
    print(f"耗時：約 {toc[4] - tic[4]:.1f} 秒")


if __name__ == "__main__":
    main()
