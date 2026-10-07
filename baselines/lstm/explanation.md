# NASA Turbofan RUL Prediction: LSTM Baseline Explanation

This document provides a step-by-step, plain English explanation of the logic inside the `lstm_multi_v2.ipynb` notebook. This notebook serves as our **baseline model**, using a Long Short-Term Memory (LSTM) network to predict the Remaining Useful Life (RUL) of aircraft engines.

---

### Cell 1: Setup and Workspace Configuration
This cell prepares the environment.
*   **Mounting Drive**: On Colab it connected the notebook to Google Drive to access the NASA datasets. Locally this cell is just a note; the data is read from `CMAPSSData/` at the repo root.
*   **Directory Setup**: It navigates to the project folder where our scripts and data reside.
*   **Verification**: It lists the files in the folder to ensure everything (data files, previous checkpoints) is in the right place before starting.

---

### Cell 2: Data Preprocessing (`preprocessing.py` logic)
This cell transforms the raw sensor readings into a format the LSTM can "read."
*   **Calculating RUL**: For training, it looks at how many cycles an engine ran before failing and calculates the "Remaining Useful Life" for every row.
*   **Piecewise Linear RUL (The 125 Cap)**: We cap the RUL at **125 cycles**. Why? Because engines don't start degrading immediately. This tells the model: "As long as the RUL is above 125, the engine is healthy. Start worrying only when it drops below that."
*   **Removing Noise**: It identifies "constant sensors" (sensors that always show the same number) and removes them because they provide no useful information about degradation.
*   **Normalization**: It scales all sensor values between **0 and 1**. This ensures that a high-pressure reading (e.g., 500 psi) doesn't "overwhelm" a low-temperature reading (e.g., 0.5 degrees) in the math.
*   **Multi-Dataset Support**: It automatically runs this process for all four NASA datasets (FD001 to FD004) and saves them as CSV files.

---

### Cell 3: Model Architecture (`model.py` logic)
This cell defines the "brain" of the system-the **LSTM Model**.
*   **The LSTM Layers**: We use two stacked layers of LSTMs. Unlike standard networks, LSTMs have "memory gates" that allow them to remember important sensor patterns from many cycles ago.
*   **Dropout**: We use a 20% dropout rate. This randomly "turns off" some neurons during training to prevent the model from memorizing the data (overfitting), forcing it to learn more general patterns.
*   **Output Layer**: A final "Linear" layer takes the LSTM's memory and turns it into a single number: the predicted RUL.
*   **Data Helpers**: This cell also contains logic to turn the 2D tables into 3D "sequences." For example, it grabs a window of **30 consecutive cycles** to make a single prediction.

---

### Cell 4: Model Training (`train.py` logic)
This is the "practice session" where the model learns from historical failures.
*   **DataLoaders**: It feeds data to the model in small batches (64 sequences at a time) rather than all at once. This makes the training faster and more stable.
*   **Learning**: It uses the **Adam optimizer** to tweak the model's internal weights. If the model guesses 50 but the answer is 40, the optimizer adjusts the "brain" to be more accurate next time.
*   **Persistence**: Once training is done, it saves the results as a **`.pth` file** (e.g., `lstm_model_FD001.pth`). This stores the "learned knowledge" so we can use the model later without having to train it again.

---

### Cell 5: Evaluation and Visualization (`evaluate.py` logic)
This is the "final exam" where we test the model on engines it has never seen before.
*   **Last-Window Protocol**: For the test set, we only give the model the **very last window** of data available for an engine. This mirrors a real-world scenario where you want to know: "Based on what I see right now, how much life is left?"
*   **Standalone Execution**: This cell is designed to be run on its own. It loads the `.pth` files from disk, meaning you can skip the 50-epoch training cell if you've already trained the model.
*   **Sorted Visualization**: To make the results easy to read, it sorts the test engines from "least life left" to "most life left."
    *   **Blue Dots**: The true engine life.
    *   **Red Stars**: What our model predicted.
    *   **Dashed Line**: The 125-cycle cap.
*   **RMSE Score**: It calculates the "Root Mean Squared Error"-the average number of cycles the model was off by. This is the main number used to compare this LSTM against our DCNN variant.
