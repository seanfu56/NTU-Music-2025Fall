#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

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

def find_mp3s(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.mp3") if p.is_file()])

def demucs_vocals_only(in_paths: List[Path], src_root: Path, out_root: Path,
                       device: str, jobs: int, segment: float):
    if not has_demucs_cli():
        raise SystemExit("Demucs 未安裝，請先：pip install demucs")

    # 轉換 device='auto' -> 'cuda' or 'cpu'
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory(prefix="demucs_out_") as tmpdir:
        tmpdir = Path(tmpdir)
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", DEMUCS_MODEL,
            "--two-stems", "vocals",
            "-d", device,                # <<< 關鍵：指定裝置
            "-j", str(jobs),             # CPU 並行數（解碼/IO）
            "-o", str(tmpdir),
            "--segment", str(int(segment)),   # 分段推論，降低 VRAM 需求（如 10~20 秒）
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
    ap = argparse.ArgumentParser(
        description="針對 data_root 底下所有 .mp3，輸出人聲至 out_dir（保留目錄結構，可用 GPU）"
    )
    ap.add_argument("--data_root", required=True, type=str, help="輸入資料夾 (遞迴搜尋 .mp3)")
    ap.add_argument("--out_dir", default='data_demucs', type=str, help="輸出資料夾")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                    help="推論裝置（預設 auto：有 GPU 則用 cuda）")
    ap.add_argument("--jobs", type=int, default=2, help="CPU 並行數（解碼/IO），GPU 不會因 jobs 提升")
    ap.add_argument("--segment", type=float, default=7.8,
                    help="分段秒數，越小越省 VRAM（10~20 是常見範圍）")
    args = ap.parse_args()

    src_root = Path(args.data_root).resolve()
    out_root = Path(args.out_dir).resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"輸入資料夾不存在：{src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    mp3s = find_mp3s(src_root)
    if not mp3s:
        print("[warn] 沒找到任何 .mp3")
        return

    print(f"共找到 {len(mp3s)} 個檔案")
    tic = os.times()
    demucs_vocals_only(mp3s, src_root, out_root, args.device, args.jobs, args.segment)
    toc = os.times()
    print(f"完成，結果保存在：{out_root}")
    print(f"耗時：約 {toc[4] - tic[4]:.1f} 秒")

if __name__ == "__main__":
    main()