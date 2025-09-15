import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from tqdm import tqdm
from sklearn.decomposition import PCA


def _features_from_y(
    y: np.ndarray,
    sr: int,
    *,
    n_mfcc: int = 20,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """Compute features from an audio array. Returns 1D vector or empty array on failure."""
    try:
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)))

        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        S_power = S**2
        mel = librosa.feature.melspectrogram(S=S_power, sr=sr)

        def _vec(a: np.ndarray) -> np.ndarray:
            return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)

        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_feat = np.concatenate([
            _vec(mfcc.mean(axis=1)), _vec(mfcc.std(axis=1)),
            _vec(mfcc_delta.mean(axis=1)), _vec(mfcc_delta.std(axis=1)),
        ])

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma_feat = np.concatenate([_vec(chroma.mean(axis=1)), _vec(chroma.std(axis=1))])

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        flatness = librosa.feature.spectral_flatness(S=S)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

        spec_stats = np.concatenate([
            _vec(centroid.mean(axis=1)), _vec(centroid.std(axis=1)),
            _vec(bandwidth.mean(axis=1)), _vec(bandwidth.std(axis=1)),
            _vec(rolloff.mean(axis=1)), _vec(rolloff.std(axis=1)),
            _vec(flatness.mean(axis=1)), _vec(flatness.std(axis=1)),
            _vec(contrast.mean(axis=1)), _vec(contrast.std(axis=1)),
            _vec(zcr.mean(axis=1)), _vec(zcr.std(axis=1)),
        ])

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        rhythm = np.array([tempo], dtype=np.float32)

        feats = np.concatenate([
            _vec(mfcc_feat), _vec(chroma_feat), _vec(spec_stats), _vec(rhythm)
        ]).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats
    except Exception as e:
        print(f"[warn] feature extraction (from y) failed: {e}")
        return np.array([], dtype=np.float32)


def extract_features(
    audio_path: str,
    sr: int = 22050,
    max_duration: float = 30.0,
    n_mfcc: int = 20,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Extract a compact set of hand-crafted features for music artist classification.

    Features (frame-wise stats collapsed to track-level):
    - MFCCs (n_mfcc): mean, std
    - Delta MFCCs: mean, std
    - Chroma CQT (12): mean, std
    - Spectral centroid/bandwidth/rolloff/flatness: mean, std
    - Spectral contrast (7): mean, std
    - Zero-crossing rate: mean, std
    - Rhythm: tempo (single value)
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if max_duration is not None:
            max_len = int(max_duration * sr)
            if len(y) > max_len:
                start = (len(y) - max_len) // 2
                y = y[start : start + max_len]
        feats = _features_from_y(y, sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        return feats
    except Exception as e:
        # If any extraction error occurs, return empty to signal skip
        print(f"[warn] feature extraction failed for {audio_path}: {e}")
        return np.array([], dtype=np.float32)


def extract_features_segments(
    audio_path: str,
    *,
    sr: int = 22050,
    segment_duration: float = 7.5,
    num_segments: int = 3,
    strategy: str = "uniform",
    n_mfcc: int = 20,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> List[np.ndarray]:
    """Extract multiple segment-level features from one track for augmentation/robustness."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"[warn] segment load failed for {audio_path}: {e}")
        return []

    seg_len = int(segment_duration * sr)
    if len(y) < max(seg_len, n_fft):
        # Too short: just compute one feature on padded full signal
        f = _features_from_y(y, sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        return [f] if f.size else []

    feats: List[np.ndarray] = []
    if strategy == "uniform":
        # Place segments evenly across the central region
        margin = max(0, (len(y) - seg_len) // (num_segments + 1))
        for i in range(num_segments):
            start = margin * (i + 1)
            end = start + seg_len
            seg = y[start:end]
            f = _features_from_y(seg, sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
            if f.size:
                feats.append(f)
    else:
        rng = np.random.default_rng(42)
        for _ in range(num_segments):
            start = int(rng.integers(0, len(y) - seg_len))
            end = start + seg_len
            seg = y[start:end]
            f = _features_from_y(seg, sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
            if f.size:
                feats.append(f)
    return feats


def _artist_from_path(p: Path) -> str:
    parts = list(p.parts)
    if "train_val" in parts:
        try:
            idx = parts.index("train_val")
            return parts[idx + 1]
        except Exception:
            pass
    return p.parent.parent.name


def load_split_list(base_dir: Path, json_name: str) -> Tuple[List[str], List[str]]:
    """Load split list from artist20/<json_name> and derive labels.

    JSON contains entries like './train_val/artist/album/track.mp3'.
    Returns (full_paths, labels).
    """
    split_path = base_dir / json_name
    with open(split_path, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    full_paths: List[str] = []
    labels: List[str] = []
    for rel in rel_list:
        print(base_dir)
        p = base_dir / rel.lstrip("./")
        print(p)
        # assert False
        if not p.exists():
            print(f"[warn] Missing file listed in {json_name}: {rel}")
            continue
        full_paths.append(str(p))
        labels.append(_artist_from_path(p))
    return full_paths, labels


def build_feature_matrix(
    file_paths: List[str],
    desc: str = "Extracting",
    *,
    segments_per_file: int | None = None,
    segment_duration: float = 7.5,
    segment_strategy: str = "uniform",
) -> Tuple[np.ndarray, List[int]]:
    """Build features.

    If segments_per_file is None, extract one vector per file. Otherwise, extract multiple
    segments per file and return repeated indices mapping each row to its source file index.
    """
    X: List[np.ndarray] = []
    kept_indices: List[int] = []
    if segments_per_file is None:
        for i, p in enumerate(tqdm(file_paths, desc=desc, ncols=80)):
            feats = extract_features(p)
            if feats.size == 0:
                continue
            kept_indices.append(i)
            X.append(feats)
    else:
        for i, p in enumerate(tqdm(file_paths, desc=desc, ncols=80)):
            seg_feats = extract_features_segments(
                p,
                segment_duration=segment_duration,
                num_segments=segments_per_file,
                strategy=segment_strategy,
            )
            for f in seg_feats:
                if f.size:
                    X.append(f)
                    kept_indices.append(i)  # repeat file index per segment
    if not X:
        raise RuntimeError("No features extracted. Check audio paths and dependencies.")
    dim = max(f.shape[0] for f in X)
    X_fixed = []
    for f in X:
        if f.shape[0] < dim:
            f = np.pad(f, (0, dim - f.shape[0]))
        elif f.shape[0] > dim:
            f = f[:dim]
        X_fixed.append(f)
    return np.vstack(X_fixed), kept_indices


def train_svm_classifier(
    X: np.ndarray,
    y: List[str],
    *,
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    use_pca: bool = False,
    pca_var: float = 0.98,
) -> Pipeline:
    """
    SVM with RBF kernel tends to work well for these compact features.
    StandardScaler is crucial for SVM performance.
    """
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_var, svd_solver="full")))
    steps.append(("svm", SVC(C=C, kernel=kernel, gamma=gamma, probability=True, class_weight=None)))
    clf = Pipeline(steps)
    clf.fit(X, y)
    return clf


def predict_topk(model: Pipeline, X: np.ndarray, label_order: List[str], k: int = 3) -> List[List[str]]:
    probs = model.predict_proba(X)
    # Prefer the estimator's class order to avoid mismatches
    try:
        classes = list(model.named_steps["svm"].classes_)
    except Exception:
        classes = label_order
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    return [[classes[j] for j in row] for row in topk_idx]


def save_label_mapping(labels: List[str], out_path: str) -> None:
    uniq = sorted(set(labels))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(uniq, f, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artist20_dir = data_root / "artist20"
    train_val_dir = artist20_dir / "train_val"
    test_dir = artist20_dir / "test"

    # 1) Load official splits from JSON (avoid album leakage)
    train_paths, y_train = load_split_list(artist20_dir, "train.json")
    val_paths, y_val = load_split_list(artist20_dir, "val.json")

    print(len(train_paths), len(val_paths))
    if not train_paths or not val_paths:
        raise FileNotFoundError("Empty train/val lists from JSON. Check data_root and JSON contents.")

    # 3) Feature extraction
    print(f"Extracting features for {len(train_paths)} train and {len(val_paths)} val tracks...")
    # Train with multiple segments per file for augmentation
    X_train, map_train = build_feature_matrix(
        train_paths,
        desc="Train segments",
        segments_per_file=args.train_segments,
        segment_duration=args.segment_duration,
        segment_strategy=args.segment_strategy,
    )
    y_train = [y_train[i] for i in map_train]
    skipped_train = len(train_paths) - len(set(map_train))
    if skipped_train:
        print(f"[info] Skipped {skipped_train} training files due to extraction failure")

    # Val with multiple segments; evaluate aggregated per file
    X_val, map_val = build_feature_matrix(
        val_paths,
        desc="Val segments",
        segments_per_file=args.eval_segments,
        segment_duration=args.segment_duration,
        segment_strategy="uniform",
    )
    skipped_val = len(val_paths) - len(set(map_val))
    if skipped_val:
        print(f"[info] Skipped {skipped_val} validation files due to extraction failure")

    # 4) Train SVM
    print("Training SVM classifier...")
    model = train_svm_classifier(
        X_train,
        y_train,
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
        use_pca=args.use_pca,
        pca_var=args.pca_var,
    )

    # 5) Evaluate (Top-1 and Top-3)
    # Train metrics (segment level, still useful for overfit check)
    y_train_pred = model.predict(X_train)
    acc_top1_train = accuracy_score(y_train, y_train_pred)
    train_probs = model.predict_proba(X_train)
    classes = list(model.named_steps["svm"].classes_)
    top3_idx_train = np.argsort(-train_probs, axis=1)[:, :3]
    top3_labels_train = [[classes[j] for j in row] for row in top3_idx_train]
    acc_top3_train = float(np.mean([yt in preds for yt, preds in zip(y_train, top3_labels_train)]))

    # Val metrics aggregated per file
    val_probs_seg = model.predict_proba(X_val)
    n_files_val = len(val_paths)
    n_classes = val_probs_seg.shape[1]
    sums = np.zeros((n_files_val, n_classes), dtype=np.float64)
    counts = np.zeros(n_files_val, dtype=np.int32)
    for seg_idx, file_idx in enumerate(map_val):
        sums[file_idx] += val_probs_seg[seg_idx]
        counts[file_idx] += 1
    counts[counts == 0] = 1
    val_probs_file = sums / counts[:, None]
    val_top1_idx = val_probs_file.argmax(axis=1)
    top3_idx = np.argsort(-val_probs_file, axis=1)[:, :3]
    classes = list(model.named_steps["svm"].classes_)
    y_val_true = [y_val[i] for i in range(n_files_val)]
    y_val_pred_file = [classes[i] for i in val_top1_idx]
    acc_top1 = accuracy_score(y_val_true, y_val_pred_file)
    top3_labels = [[classes[j] for j in row] for row in top3_idx]
    acc_top3 = float(np.mean([yt in preds for yt, preds in zip(y_val_true, top3_labels)]))

    print(f"Train Top-1 Acc: {acc_top1_train:.4f}")
    print(f"Train Top-3 Acc: {acc_top3_train:.4f}")
    print(f"Validation Top-1 Acc: {acc_top1:.4f}")
    print(f"Validation Top-3 Acc: {acc_top3:.4f}")
    try:
        print(classification_report(y_val, y_val_pred))
    except Exception:
        pass

    # 6) Optional sanity check: shuffled labels training
    if args.sanity_shuffle_labels:
        rng = np.random.default_rng(0)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        print("\n[Sanity] Training another SVM on shuffled labels...")
        model_shuf = train_svm_classifier(
            X_train,
            y_shuf,
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            use_pca=args.use_pca,
            pca_var=args.pca_var,
        )
        acc_train_shuf = accuracy_score(y_shuf, model_shuf.predict(X_train))
        print(f"[Sanity] Train acc with shuffled labels: {acc_train_shuf:.4f} (should be near random, ~{1/len(set(y_train)):.3f})")

    # 7) Persist artifacts
    label_list = sorted(set(y_train + y_val))
    dump(model, output_dir / "svm_artist_model.joblib")
    save_label_mapping(label_list, str(output_dir / "labels.json"))
    print(f"Saved model and labels to {output_dir}")

    # 8) Predict test top-3 to match grader JSON format
    test_files = sorted([p for p in Path(test_dir).glob("*.mp3")])
    if test_files:
        print(f"Extracting features for {len(test_files)} test tracks...")
        X_test, map_test = build_feature_matrix(
            [str(p) for p in test_files],
            desc="Test segments",
            segments_per_file=args.eval_segments,
            segment_duration=args.segment_duration,
            segment_strategy="uniform",
        )
        # Aggregate per test file
        probs_seg = model.predict_proba(X_test)
        n_files = len(test_files)
        n_classes = probs_seg.shape[1]
        sums = np.zeros((n_files, n_classes), dtype=np.float64)
        counts = np.zeros(n_files, dtype=np.int32)
        for seg_idx, file_idx in enumerate(map_test):
            sums[file_idx] += probs_seg[seg_idx]
            counts[file_idx] += 1
        counts[counts == 0] = 1
        probs_file = sums / counts[:, None]
        top3_idx = np.argsort(-probs_file, axis=1)[:, :3]
        classes = list(model.named_steps["svm"].classes_)
        top3 = [[classes[j] for j in row] for row in top3_idx]
        pred_dict: Dict[str, List[str]] = {}
        for p, preds in zip(test_files, top3):
            key = p.stem  # e.g., '001'
            pred_dict[key] = preds
        with open(output_dir / "test_pred.json", "w", encoding="utf-8") as f:
            json.dump(pred_dict, f, ensure_ascii=False, indent=2)
        print(f"Wrote predictions to {output_dir / 'test_pred.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional ML SVM Artist Classifier")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder containing artist20/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory to write model and predictions",
    )
    parser.add_argument("--C", type=float, default=1.0, help="SVM C (regularization strength)")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear", "poly", "sigmoid"], help="SVM kernel")
    parser.add_argument("--gamma", type=str, default="auto", help="SVM gamma (for rbf/poly/sigmoid), 'scale' or 'auto' or float")
    parser.add_argument("--use_pca", action="store_true", default=True, help="Insert PCA before SVM for regularization")
    parser.add_argument("--pca_var", type=float, default=0.95, help="PCA retained variance if --use_pca")
    parser.add_argument("--segment_duration", type=float, default=7.5, help="Segment duration in seconds for segment-based features")
    parser.add_argument("--train_segments", type=int, default=3, help="#segments per track for training")
    parser.add_argument("--eval_segments", type=int, default=5, help="#segments per track for val/test aggregation")
    parser.add_argument("--segment_strategy", type=str, default="uniform", choices=["uniform", "random"], help="How to pick training segments")
    parser.add_argument("--sanity_shuffle_labels", action="store_true", help="Train an additional model with shuffled labels to detect leakage")
    args = parser.parse_args()
    # Coerce gamma to float if numeric string was provided
    try:
        if isinstance(args.gamma, str) and args.gamma not in ("scale", "auto"):
            args.gamma = float(args.gamma)
    except Exception:
        args.gamma = "scale"
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)
