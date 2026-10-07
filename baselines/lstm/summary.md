# Project Summary: NASA C-MAPSS Turbofan RUL Prediction (LSTM Baseline)

**Last Updated**: 2026-05-02
**Current Phase**: Finalized

## 🎯 Project Objective
Predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset. This notebook serves as the **baseline model**, using a two-layer unidirectional LSTM to establish a performance benchmark across all four C-MAPSS subsets (FD001–FD004).

## 🛠️ Environment Setup
- **Platform**: Google Colab (GPU)
- **Core Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `matplotlib`: Visualization
  - `scikit-learn`: Preprocessing & RMSE calculation
  - `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`: Deep Learning (PyTorch)

## 📊 Datasets
The NASA C-MAPSS data is divided into three key file types for each subset (e.g., FD001):

### 1. `train_FD00x.txt` (Full Life Data)
- **Content**: Complete sensor history for multiple engines from start-of-life until **failure**.
- **RUL Calculation**: RUL = (MaxCycle_Unit) - (CurrentCycle), clipped at a maximum of **125 cycles**.

### 2. `test_FD00x.txt` (Partial Life Data)
- **Content**: Sensor history for a new set of engines, stopping before failure.
- **Goal**: The model uses only the **last window** of each engine's test data to predict its remaining life.

### 3. `RUL_FD00x.txt` (Ground Truth Answer Key)
- **Content**: One number per test engine, the **additional cycles** remaining after the test data ends.
- **RUL Calculation**: RUL_Current = (ValueInRULFile) + (MaxCycle_UnitInTestFile) - (CurrentCycle)

## 🧠 Model Architecture (`model.py` cell)
The model is a standard **unidirectional LSTM** with the following configuration:
- **Input**: A sliding window of **30 consecutive cycles**, each with the selected sensor features.
- **LSTM Layers**: 2 stacked LSTM layers, `HIDDEN_SIZE = 128`.
- **Regularization**: `Dropout(0.2)` applied after the final LSTM step.
- **Output**: A single fully-connected layer producing one scalar (the predicted RUL).
- **Dynamic Input Sizing**: The model automatically adapts to each dataset's number of non-constant sensors.

The `model.py` cell also defines two sequence-building helpers:
- `create_sequences`: Creates all sliding windows for training.
- `create_last_sequences`: Creates **only the final window** per engine, used for standardized test evaluation.

## 🔄 Preprocessing (`preprocessing.py` cell)
- **Sensor Selection**: Identifies and removes constant-value sensors per-dataset automatically.
- **Piecewise Linear RUL**: Labels clipped at **125 cycles** (focuses on the degradation phase).
- **Normalization**: Min-Max Scaler fitted on training data only, then applied to test data (no leakage). Scale range: **[0, 1]**.
- **Multi-Dataset**: Loops through FD001–FD004. Saved CSVs: `train_preprocessed_FD00x.csv`, `test_preprocessed_FD00x.csv`.

## 🏋️ Training (`train.py` cell)
- **Loss Function**: `MSELoss` (numerically stable; final RMSE computed from it).
- **Optimizer**: `Adam`, `learning_rate = 0.001`.
- **Epochs**: 50 per dataset.
- **Batch Size**: 64, using `DataLoader` with shuffling.
- **Checkpointing**: Saves `lstm_model_FD00x.pth` with model weights, `input_size`, `hidden_size`, `num_layers`, and `window_size` for later standalone evaluation.

## ✅ Evaluation Protocol (`evaluate.py` cell)
The evaluation cell is **completely standalone**, it can be run after a kernel restart without re-running training:
1. Loads the appropriate `lstm_model_FD00x.pth` checkpoint from disk.
2. Reconstructs the `LSTMModel` using the saved architecture metadata.
3. Builds the test set using `create_last_sequences`, **one prediction per engine, from its final window only**. This matches the standard C-MAPSS evaluation protocol used by the DCNN paper.
4. Runs inference in batches via `DataLoader` (avoids CUDA OOM errors on large datasets like FD002/FD004).
5. Computes and prints the final **Test RMSE**.
6. Generates the **sorted RUL visualization** described below.

> **Note**: Before running the evaluation cell independently, you must first run the `model.py` cell to load the `LSTMModel` class and `create_last_sequences` function into memory.

## 📈 Visualization
The evaluation cell produces a "paper-style" sorted prediction plot for each dataset:
- **X-axis**: Test engines sorted by their true RUL (ascending).
- **Blue Line + Dots**: Ground truth (Actual RUL).
- **Red Line + Stars**: Model predictions.
- **Dashed Black Line**: Rectified RUL cap at 125 cycles.
- **Title**: Includes the dataset ID and computed RMSE (e.g., `"Sorted Prediction for Testing Engine Units (FD001 - RMSE: 13.41)"`).

This format matches the visualization style used in the Li et al. (2018) DCNN paper, enabling direct visual comparison between the LSTM baseline and DCNN variant.

## 📂 Key Files
- `lstm_multi_v2.ipynb`: **The finalized notebook**, runs the full pipeline (Preprocessing → Model → Training → Evaluation) for all four datasets.
- `lstm_model_{ID}.pth`: Saved checkpoints per dataset. Used for standalone evaluation.
- `train_preprocessed_FD00x.csv` / `test_preprocessed_FD00x.csv`: Preprocessed data CSVs.
- `preprocessing.py`, `train.py`, `evaluate.py`, `model.py`: Standalone script versions of each cell.
- `summary.md`: This file.

## 📊 Final Results

| Dataset | Test RMSE |
|---------|-----------|
| FD001   | ~13.41    |
| FD002   | ~20.81    |
| FD003   | ~12.31    |
| FD004   | ~24.64    |
