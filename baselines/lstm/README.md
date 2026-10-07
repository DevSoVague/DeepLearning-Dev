# LSTM Baseline - Turbofan RUL Prediction

This folder contains the **Baseline** variant using a Long Short-Term Memory (LSTM) network for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

## Overview
The `lstm_multi_v2.ipynb` notebook implements a robust recurrent neural network pipeline. It is designed to capture temporal dependencies in sensor data, providing a benchmark against which more complex architectures (like the DCNN) can be measured.

---

## 1. Data Preprocessing (`preprocessing.py` logic)
The preprocessing stage cleans and structures the raw C-MAPSS data for time-series modeling:

*   **Dynamic Sensor Selection:** Automatically identifies and removes sensors that remain constant over time, ensuring the model only trains on informative signals.
*   **Piecewise Linear RUL:** Implements the "healthy-to-degrading" strategy by capping RUL at 125 cycles. This forces the model to learn the specific signatures of engine wear rather than estimating life during the stable "healthy" phase.
*   **Normalization:** Scales sensor values between 0 and 1 using a Min-Max Scaler, fitted strictly on training data to prevent any data leakage from the test sets.
*   **Sliding Window Generation:** Transforms 2D dataframes into 3D sequences (30-cycle windows) suitable for recurrent processing.
*   **Target Labeling:** Maps the RUL of the final cycle in each window as the ground truth label for the entire sequence.

## 2. Model Architecture (`model.py` logic)
The baseline uses a **Stacked LSTM** network designed for sequence regression:

*   **Recurrent Layers:** Two layers of LSTMs with 128 hidden units each. These layers "remember" patterns across the 30-cycle window, tracking how sensor values drift as the engine degrades.
*   **Regularization:** Incorporates a 20% **Dropout** rate after the recurrent layers to improve generalization and prevent the model from overfitting to specific engine units.
*   **Regression Tail:** The final hidden state is passed through a fully connected (Linear) layer to produce a single continuous value representing the predicted RUL.
*   **Sequence Helpers:** Includes specialized logic (`create_last_sequences`) to extract only the final snapshot of each test engine, ensuring evaluation follows the standard NASA protocol.

## 3. Training Process (`train.py` logic)
The training phase optimizes the network weights using historical engine failures:

*   **Loss Function:** Utilizes **MSE (Mean Squared Error)** for numerical stability. The final performance is reported as **RMSE** (the square root of the loss).
*   **Optimization:** Uses the **Adam** optimizer with a learning rate of 0.001 and a batch size of 64.
*   **Batch Processing:** Employs PyTorch `DataLoaders` to handle the large multi-dataset volumes (especially FD002 and FD004) without exhausting system memory.
*   **Persistence:** Automatically exports trained weights as `.pth` files (e.g., `lstm_model_FD001.pth`), saving the model's architecture metadata along with its learned parameters.

## 4. Evaluation and Visualization (`evaluate.py` logic)
The final stage provides a rigorous assessment of the model's predictive accuracy:

*   **Standalone Execution:** The evaluation cell is fully decoupled from training. It reconstructs the model and loads saved weights from disk, allowing for immediate performance checks on a fresh kernel.
*   **Sorted Visualization:** Generates plots where test engines are sorted by their actual RUL. This creates a smooth "descending staircase" that makes it visually obvious where the model is over-predicting or under-predicting.
*   **Visual Diagnosis:**
    *   **Blue Dots/Line:** Ground truth (Actual RUL).
    *   **Red Stars/Line:** LSTM predictions.
    *   **Black Dashed Line:** The 125-cycle rectification cap.
*   **Performance Metrics:** Reports final RMSE for each of the four datasets (FD001 - FD004), with results displayed clearly in both the console and the plot titles.
