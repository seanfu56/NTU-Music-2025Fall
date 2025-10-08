#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser("Evaluate Top-1/Top-3 and plot percentage confusion matrix (no seaborn)")
    ap.add_argument("pred_json", type=str, help="Path to predictions.json")
    # 真實標籤的萃取設定（預設：key.split('/')[1]）
    ap.add_argument("--split-delim", type=str, default="/", help="Delimiter to split prediction keys")
    ap.add_argument("--gt-pos", type=int, default=1, help="Index of ground-truth label after splitting key")
    # 影像輸出
    ap.add_argument("--title", type=str, default="Confusion Matrix (row-normalized, %)")
    ap.add_argument("--save", type=str, default=None, help="Path to save the confusion matrix image (e.g., cm.png)")
    ap.add_argument("--figsize", type=str, default="10,8", help="Figure size W,H in inches")
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving")
    ap.add_argument("--fontsize", type=int, default=9, help="Base font size for ticks/labels")
    # 是否只用出現在 y_true 裡的標籤（避免極長類別清單）
    ap.add_argument("--labels-from", type=str, default="union", choices=["union", "ytrue"],
                    help="Build label set from 'union' of y_true & y_pred or only from y_true")
    return ap.parse_args()


def load_predictions(pred_path: Path) -> Dict[str, List[str]]:
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    # 允許 list[dict] 或 dict 但標準是 dict
    if isinstance(preds, list):
        raise SystemExit("Unsupported JSON format: expected {key: [topk_labels...]}, got a list.")
    return preds


def extract_gt(key: str, delim: str, pos: int) -> str:
    parts = key.split(delim)
    if len(parts) <= pos:
        raise ValueError(f"Key '{key}' cannot provide ground truth with split '{delim}' and index {pos}.")
    return parts[pos]


def build_confusion(y_true_names: List[str], y_pred_names: List[str], labels: List[str]) -> np.ndarray:
    """Return raw count confusion matrix with given label order."""
    idx_map = {name: i for i, name in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for yt, yp in zip(y_true_names, y_pred_names):
        if yt not in idx_map or yp not in idx_map:
            # 跳過未知標籤（理論上不會發生，除非 labels 過濾掉了）
            continue
        cm[idx_map[yt], idx_map[yp]] += 1
    return cm


def plot_confusion_percentage(cm_counts: np.ndarray, labels: List[str],
                              title: str, figsize: Tuple[float, float], dpi: int, fontsize: int,
                              save_path='confusion_matrix.png'):
    # 按列（真實類別）做 row-normalize -> 百分比
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_perc = (cm_counts / safe_row_sums) * 100.0  # 每列加總=100%

    # 畫圖（純 matplotlib）
    plt.figure(figsize=figsize, dpi=dpi)
    im = plt.imshow(cm_perc, interpolation="nearest", aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("Percentage (%)", fontsize=fontsize)

    num_classes = len(labels)
    plt.xticks(ticks=np.arange(num_classes), labels=labels, rotation=45, ha="right", fontsize=fontsize)
    plt.yticks(ticks=np.arange(num_classes), labels=labels, fontsize=fontsize)

    plt.xlabel("Predicted Label", fontsize=fontsize + 1)
    plt.ylabel("True Label", fontsize=fontsize + 1)
    plt.title(title, fontsize=fontsize + 2)

    # 在每個 cell 上標註百分比
    thresh = cm_perc.max() * 0.5  # 深/淺色切換文字顏色
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_perc[i, j]
            txt_color = "white" if val >= thresh else "black"
            plt.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=fontsize)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[info] Confusion matrix saved to: {save_path}")
    plt.show()


def main():
    args = parse_args()
    pred_path = Path(args.pred_json)
    if not pred_path.exists():
        print(f"File not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    preds = load_predictions(pred_path)

    y_true_names: List[str] = []
    y_pred_names: List[str] = []
    top1_correct = 0
    top3_correct = 0
    total = 0

    # 讀取並計算
    for k, pred_list in preds.items():
        if not isinstance(pred_list, list) or len(pred_list) == 0:
            print(f"[warn] key={k} has empty or invalid prediction list; skip")
            continue

        try:
            gt = extract_gt(k, args.split_delim, args.gt_pos)
        except ValueError as e:
            print(f"[warn] {e}; skip")
            continue

        total += 1
        top1 = pred_list[0]
        if gt == top1:
            top1_correct += 1
            top3_correct += 1
        elif gt in pred_list[:3]:
            top3_correct += 1

        y_true_names.append(gt)
        y_pred_names.append(top1)

    if total == 0:
        print("No valid samples parsed. Check --split-delim/--gt-pos to extract ground truth correctly.")
        sys.exit(1)

    acc1 = 100.0 * top1_correct / total
    acc3 = 100.0 * top3_correct / total
    print(f"Top-1 accuracy: {acc1:.2f}%")
    print(f"Top-3 accuracy: {acc3:.2f}%")

    # 準備標籤順序
    if args.labels_from == "ytrue":
        labels = sorted(set(y_true_names))
    else:
        labels = sorted(set(y_true_names) | set(y_pred_names))

    # 建 raw confusion，然後畫「百分比」版本
    cm_counts = build_confusion(y_true_names, y_pred_names, labels)

    # 解析 figsize
    try:
        w_str, h_str = args.figsize.split(",")
        figsize = (float(w_str), float(h_str))
    except Exception:
        figsize = (10.0, 8.0)

    save_path = 'confusion_matrix.png'
    plot_confusion_percentage(
        cm_counts, labels,
        title=args.title,
        figsize=figsize,
        dpi=int(args.dpi),
        fontsize=int(args.fontsize),
        save_path=save_path
    )


if __name__ == "__main__":
    main()