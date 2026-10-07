# LSTM Baseline vs. Deep CNN Variant: Final Results Comparison

> **C-MAPSS Engine Degradation Dataset (NASA Turbofan)**
> Fair, like-for-like evaluation using the **last-window-only** protocol
> (only the final observed window per engine is used for test prediction).

---

## Model Architectures

| Feature | LSTM Baseline | Deep CNN Variant |
|---|---|---|
| **Architecture** | 2-layer unidirectional LSTM | 5-layer 2D Deep CNN |
| **Hidden/Filter Size** | 128 hidden units | 10 filters (conv1–4), 1 filter (conv5) |
| **FC Layers** | 1 FC layer (128 → 1) | 2 FC layers (flattened → 100 → 1) |
| **Dropout** | 0.2 | 0.5 |
| **Activation** |, | Tanh throughout |
| **Weight Init** | Default (PyTorch) | Xavier Normal |
| **Loss Function** | MSE | RMSE (custom, with ε) |
| **Optimizer** | Adam (lr=0.001) | Adam (lr=0.001) |
| **LR Schedule** | None | Drop to 0.0001 at epoch 200 |
| **Batch Size** | 64 | 512 |
| **Epochs** | 50 | 250 |

---

## Preprocessing Differences

| Feature | LSTM Baseline | Deep CNN Variant |
|---|---|---|
| **Sensor Selection** | All non-constant sensors (via MinMaxScaler) | 14 selected sensors (paper-defined) |
| **Normalization Range** | [0, 1] | [−1, 1] |
| **Scaler Fit** | On training data only (no leakage) | On training data only (no leakage) |
| **RUL Clipping** | 125 | 125 |
| **Test Evaluation** | Last window per engine only | Last window per engine only |

---

## Window Sizes

| Dataset | LSTM | Deep CNN |
|---|---|---|
| **FD001** | 30 | 30 |
| **FD002** | 30 | 20 |
| **FD003** | 30 | 30 |
| **FD004** | 30 | 15 |

---

## Test RMSE Results (Lower is Better)

| Dataset | LSTM Baseline | Deep CNN Variant | Δ RMSE | Winner |
|---|---|---|---|---|
| **FD001** | 13.41 | 13.14 | −0.27 | **DCNN** ✅ |
| **FD002** | 20.81 | 21.04 | +0.23 | **LSTM** ✅ |
| **FD003** | 12.31 | 11.98 | −0.33 | **DCNN** ✅ |
| **FD004** | 24.64 | 27.10 | +2.46 | **LSTM** ✅ |
| **Average** | **17.79** | **18.32** | +0.52 | **LSTM** ✅ |

---

## Deep CNN Score Results (NASA Asymmetric Scoring, Lower is Better)

> Note: The LSTM notebook does not report the NASA Score metric.
> The asymmetric scoring function penalizes **late predictions** (d > 0) more than
> **early predictions** (d < 0): Score = Σ (exp(d/10)−1) if d>0, else Σ (exp(−d/13)−1).

| Dataset | Deep CNN Score |
|---|---|
| **FD001** | 311.05 |
| **FD002** | 4,620.68 |
| **FD003** | 359.08 |
| **FD004** | 10,844.59 |
| **Total** | **16,135.40** |

---

## Key Takeaways

1. **Results are extremely close overall.** Both models achieve similar RMSE across all four datasets, with margins of less than 0.3 RMSE on FD001 and FD003.

2. **DCNN wins on single-condition datasets (FD001, FD003).** These datasets have one operating condition, which plays to the DCNN's strength of learning spatial/convolutional feature maps.

3. **LSTM wins on multi-condition datasets (FD002, FD004).** These datasets have multiple operating conditions. The LSTM's temporal memory appears to generalize better, especially on FD004 where the DCNN trails by ~2.46 RMSE.

4. **The evaluation protocol matters enormously.** Both models now use the same "last-window-only" test protocol, enabling a genuine apples-to-apples comparison. Earlier LSTM evaluations (using all sliding windows) inflated its scores artificially.

5. **The DCNN uses significantly more training resources** (250 epochs vs. 50, 5× larger batch size), but doesn't consistently outperform the simpler LSTM across all datasets.

6. **LSTM is the stronger general-purpose baseline** when averaging across all four datasets (RMSE 17.79 vs. 18.32), despite using far fewer training epochs.

---

## References

- LSTM Baseline: `lstm/lstm_multi_v2.ipynb`
- Deep CNN Variant: `dcnn/final_dcnn_variant.ipynb`
- Evaluation Protocol: Last-window-only (matches standard C-MAPSS convention used by Babu et al., 2016)
