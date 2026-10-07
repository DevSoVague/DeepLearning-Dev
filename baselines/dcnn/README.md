# Final DCNN Variant - Turbofan RUL Prediction

This folder contains the **Final** variant of the Deep Convolutional Neural Network (DCNN) for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

## Overview
The `final_dcnn_variant.ipynb` notebook implements a state-of-the-art predictive maintenance pipeline. It processes time-series sensor data to estimate how many cycles an engine has left before failure.

---

## 1. Data Preprocessing (`preprocessing.py` logic)
The preprocessing stage transforms raw sensor data into a format suitable for deep learning:

*   **Sensor Selection:** Focuses on 14 critical sensors as recommended by literature, ignoring noise-heavy sensors.
*   **Piecewise RUL:** Implements a "healthy-to-degrading" model where RUL is capped at 125 cycles, helping the model focus on the actual degradation phase.
*   **Normalization:** Scales sensor values between -1 and 1 using a Min-Max Scaler (fitted only on training data to prevent leakage).
*   **Sliding Window Generation:** Converts 2D sensor tables into 3D "cubes" (sequences) of data (e.g., 30-cycle windows).
*   **Target Labeling:** Assigns the RUL of the final cycle in a window as the target for the model to predict.

## 2. Model Architecture (`model.py` logic)
The system uses a **Deep 2D Convolutional Neural Network** designed to extract temporal features:

*   **Feature Extraction:** Four convolutional layers with 10x1 filters scan the timeline of each sensor independently to find wear-and-tear signatures.
*   **Data Condensing:** A final 3x1 convolutional layer summarizes the extracted features.
*   **Activations:** Uses the `Tanh` function throughout to learn complex, non-linear degradation patterns.
*   **Regularization:** Employs a 50% **Dropout** layer to prevent the model from over-fitting or memorizing specific engines.
*   **Regression Tail:** Flattens the features and passes them through a hidden layer of 100 neurons before outputting a single RUL prediction.

## 3. Training Process (`train.py` logic)
The training phase is the "practice session" where the model learns from historical data:

*   **Loss Functions:** Uses **RMSE** for mathematical accuracy and a custom **C-MAPSS Score** which penalizes "late" predictions (over-estimating engine life) more heavily than "early" ones.
*   **Optimization:** Uses the **Adam** optimizer with a learning rate of 0.001.
*   **Learning Rate Scheduling:** Automatically drops the learning rate by 10x after 200 rounds (epochs) to fine-tune the model's accuracy.
*   **Persistence:** Saves the final trained weights as `.pth` files (e.g., `dcnn_model_FD001.pth`) for future use.

## 4. Evaluation and Visualization (`evaluate.py` logic)
The final stage provides the "Report Card" for the model's performance:

*   **Standalone Execution:** The evaluation cell now loads saved `.pth` model weights directly from disk, allowing for immediate inference without retraining.
*   **Sorted Visualization:** Plots Actual RUL vs. Predicted RUL. The data is sorted by RUL to create a "descending staircase" visual, making it easy to see where the model deviates from the truth.
*   **Visual Diagnosis:**
    *   **Blue Dots/Line:** Ground truth (Actual RUL).
    *   **Red Stars/Line:** Model predictions.
*   **Performance Metrics:** Re-calculates and displays final RMSE and Scores for each of the four NASA datasets (FD001 - FD004), embedding the dataset ID and RMSE directly into each plot's title.
