#!/usr/bin/env python3
# model_summary.py
import argparse
import torch
import torch.nn as nn

# 改成你的檔名：例如 from eval_transformer import CNNTransformer, N_MELS
from eval_transformer import CNNTransformer, N_MELS  # ← 修改這行以符合你的檔名

def build_model(n_classes: int,
                depth: int = 6, base_channels: int = 64,
                d_model: int = 256, n_heads: int = 8,
                n_layers: int = 6, ff_dim: int = 512,
                dropout: float = 0.1, device: str = "cpu") -> nn.Module:
    model = CNNTransformer(
        n_mels=N_MELS, n_classes=n_classes,
        depth=depth, base_channels=base_channels,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ff_dim=ff_dim, dropout=dropout
    ).to(device)
    return model

def main():
    ap = argparse.ArgumentParser(description="Print CNN+Transformer summary with torchinfo / torchsummary")
    ap.add_argument("--n_classes", type=int, default=20)
    ap.add_argument("--frames", type=int, default=400, help="time steps T (≈ chunk_sec / 0.01；4s→400)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_torchsummary", action="store_true", help="Use legacy torchsummary instead of torchinfo")
    # 可選：模型結構參數
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--tx_layers", type=int, default=8)
    ap.add_argument("--tx_heads", type=int, default=8)
    ap.add_argument("--tx_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(
        n_classes=args.n_classes,
        depth=args.depth, base_channels=args.base_channels,
        d_model=args.d_model, n_heads=args.tx_heads,
        n_layers=args.tx_layers, ff_dim=args.tx_ff,
        dropout=args.dropout, device=device.type
    )

    # --- 先做一次 dummy forward：讓動態 Linear `proj` 建起來 ---
    dummy = torch.randn(1, 1, N_MELS, args.frames, device=device)
    with torch.no_grad():
        _ = model(dummy)


    from torchinfo import summary
    print("\n=== torchinfo ===")
    print(summary(
        model,
        input_size=(1, 1, N_MELS, args.frames),  # (B,C,M,T)
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=2,
        device=device
    ))

if __name__ == "__main__":
    main()