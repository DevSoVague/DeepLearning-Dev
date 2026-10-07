"""
Evaluate the shipped v12 checkpoints on the C-MAPSS test sets without retraining.

Reproduces the preprocessing of dev_contri_1_2_final.ipynb (drop 7 flat sensors,
per-sensor linear regression on the 3 operating settings fitted on all four
training sets, test settings clamped to the training range, z-scored residuals,
last W=30 window per test engine) and prints test RMSE and PHM score per subset.

    python scripts/evaluate.py                         # all 6 checkpoints, data in ./CMAPSSData
    python scripts/evaluate.py --data-dir /path/to/CMAPSSData --models base_transformer
    python scripts/evaluate.py --check-only            # just load every checkpoint (no data needed)
    python scripts/evaluate.py --compare               # also diff against v12_results.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_model  # noqa: E402

COLS = (["unit", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"s_{i}" for i in range(1, 22)])
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
USE_SENSORS = [c for c in COLS if c.startswith("s_") and c not in DROP_SENSORS]
SETTINGS = ["setting_1", "setting_2", "setting_3"]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
W = 30

# checkpoint stem -> (architecture, is GRL wrapper, key in v12_results.json)
CHECKPOINTS = {
    "base_tcn":            ("tcn",         False, ("base", "TCN")),
    "base_transformer":    ("transformer", False, ("base", "Transformer")),
    "base_tcn_gru":        ("tcn_gru",     False, ("base", "TCN-GRU")),
    "grl_tcn_grl":         ("tcn",         True,  ("grl", "TCN+GRL")),
    "grl_transformer_grl": ("transformer", True,  ("grl", "Transformer+GRL")),
    "grl_tcn_gru_grl":     ("tcn_gru",     True,  ("grl", "TCN-GRU+GRL")),
}


def load_checkpoint(stem, ckpt_dir):
    arch, grl, _ = CHECKPOINTS[stem]
    model = build_model(arch, grl=grl)
    ckpt = torch.load(ckpt_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)  # raises if any key or shape differs
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "model"} if isinstance(ckpt, dict) else {}
    return model, meta


# ── Data and preprocessing (same steps as the notebook) ──────────────────────
def load_subset(data_dir, subset):
    import pandas as pd
    read = lambda name, names: pd.read_csv(data_dir / name, sep=r"\s+", header=None,
                                           names=names, engine="python")
    train = read(f"train_{subset}.txt", COLS)
    test = read(f"test_{subset}.txt", COLS)
    rul = read(f"RUL_{subset}.txt", ["RUL"])["RUL"].values
    return train, test, rul


def prepare_test_windows(data_dir):
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    raw = {s: load_subset(data_dir, s) for s in SUBSETS}
    combined = pd.concat([raw[s][0][SETTINGS + USE_SENSORS] for s in SUBSETS], ignore_index=True)
    X = combined[SETTINGS].values
    bounds = (X.min(axis=0), X.max(axis=0))
    regs = {c: LinearRegression().fit(X, combined[c].values) for c in USE_SENSORS}

    def residual(df):
        df = df.copy()
        Xs = np.clip(df[SETTINGS].values.astype(np.float64), bounds[0], bounds[1])
        for c in USE_SENSORS:
            df[c] = df[c].values - regs[c].predict(Xs)
        return df

    resid_train = pd.concat([residual(raw[s][0]) for s in SUBSETS], ignore_index=True)
    means = resid_train[USE_SENSORS].mean()
    stds = resid_train[USE_SENSORS].std().replace(0, 1.0)

    out = {}
    for s in SUBSETS:
        test = residual(raw[s][1])
        test[USE_SENSORS] = (test[USE_SENSORS] - means) / stds
        windows = []
        for _, eng in test.groupby("unit"):
            arr = eng[USE_SENSORS].values.astype(np.float32)
            if len(arr) < W:
                arr = np.concatenate([np.tile(arr[0:1], (W - len(arr), 1)), arr], 0)
            windows.append(arr[-W:])
        out[s] = (np.stack(windows), raw[s][2].astype(np.float32))
    return out


# ── Metrics ──────────────────────────────────────────────────────────────────
def phm_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1).sum())


@torch.no_grad()
def predict(model, X, batch_size=256):
    fn = model.predict_rul if hasattr(model, "predict_rul") else model
    preds = [fn(torch.from_numpy(X[i:i + batch_size])).numpy()
             for i in range(0, len(X), batch_size)]
    return np.concatenate(preds)


def evaluate(model, data):
    res = {}
    for s, (X, y) in data.items():
        p = np.clip(predict(model, X), 0, None)
        res[s] = {"rmse": math.sqrt(float(np.mean((p - y) ** 2))), "score": phm_score(y, p),
                  "mae": float(np.mean(np.abs(p - y))), "bias": float((p - y).mean())}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "CMAPSSData"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "checkpoints_v12"))
    ap.add_argument("--models", nargs="+", default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    ap.add_argument("--check-only", action="store_true", help="load checkpoints only, no data")
    ap.add_argument("--compare", action="store_true", help="compare with v12_results.json")
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    models = {}
    for stem in args.models:
        model, meta = load_checkpoint(stem, ckpt_dir)
        n = sum(p.numel() for p in model.parameters())
        extra = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in meta.items())
        print(f"loaded {stem:22s} {n:>7,} params  state_dict keys match  ({extra})")
        models[stem] = model
    if args.check_only:
        return

    data_dir = Path(args.data_dir)
    if not (data_dir / "train_FD001.txt").exists():
        sys.exit(f"C-MAPSS files not found in {data_dir} (see README, Data section).")
    data = prepare_test_windows(data_dir)

    reference = {}
    ref_path = ckpt_dir / "v12_results.json"
    if args.compare and ref_path.exists():
        reference = json.loads(ref_path.read_text())

    print(f"\nTest RMSE (PHM score), last window per engine, W={W}")
    print(f"{'model':22s} " + "  ".join(f"{s:>17s}" for s in SUBSETS))
    max_diff = 0.0
    for stem, model in models.items():
        res = evaluate(model, data)
        print(f"{stem:22s} " + "  ".join(f"{res[s]['rmse']:7.2f} ({res[s]['score']:>7.0f})" for s in SUBSETS))
        if reference:
            group, key = CHECKPOINTS[stem][2]
            ref = reference.get(group, {}).get(key)
            if ref:
                diffs = [abs(res[s]["rmse"] - ref[s]["rmse"]) for s in SUBSETS]
                max_diff = max(max_diff, *diffs)
                print(f"{'  saved (v12_results)':22s} " + "  ".join(
                    f"{ref[s]['rmse']:7.2f} ({ref[s]['score']:>7.0f})" for s in SUBSETS))
    if reference:
        print(f"\nmax |RMSE - saved RMSE| across models and subsets: {max_diff:.4f}")


if __name__ == "__main__":
    main()
