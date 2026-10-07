
### `preprocessing.py`
The `preprocessing.py` cell in your notebook is responsible for transforming the raw NASA turbofan engine data into a format that the Deep CNN model can understand. This involves cleaning the data, calculating labels, and "windowing" the time-series data.

Here is a step-by-step explanation of what the code is doing:

### 1. Setup and Feature Selection
*   **Sensor Selection:** The code ignores several sensors that typically show no useful trend and selects **14 specific sensors** (e.g., s2, s3, s4, etc.) as recommended by the research paper this notebook follows.
*   **Defining the RUL Limit (`MAX_RUL`):** It sets a threshold of **125 cycles**. This follows the "Piecewise Linear Degradation" model. The theory is that an engine's health doesn't drop immediately; it stays "healthy" (at 125) for a while before starting to degrade linearly toward zero.

### 2. Piecewise RUL Calculation (`add_piecewise_rul`)
*   For every engine in the training set, the code calculates how many cycles it has left before failure (RUL).
*   If an engine has 200 cycles left, the code "clips" that value to **125**. As the engine gets closer to failure (e.g., 50 cycles left), the value stays at 50. This helps the model focus on the period when the engine is actually showing signs of wear.

### 3. Normalization (Scaling)
*   Sensor values have different units (temperature, pressure, speed). The code uses a **Min-Max Scaler** to squash all values into a range between **-1 and 1**.
*   **Crucial Step:** The scaler is "fitted" only on the **training data** and then applied to the test data. This prevents "data leakage" (the model shouldn't know the maximum values of the test set beforehand).

### 4. Sliding Window Sequence Generation (`generate_sequences`)
Deep CNNs require a "snapshot" of time rather than a single row of data.
*   The code takes a **sequence length** (e.g., 30 cycles).
*   It slides a window across the engine's history. For example, cycles 1-30 become the first input, cycles 2-31 become the second, and so on.
*   This converts your 2D table into a 3D "cube" of data: `(Number of Samples, Time Steps, Number of Sensors)`.

### 5. Label Assignment
*   For every 30-cycle window created above, the code looks at the **very last cycle** in that window and assigns its RUL value as the "target" or "answer" for the model to predict.

### 6. Test Data Handling
*   The test set is handled differently. Since we want to predict the RUL at the "current" moment in the test set, the code extracts only the **last window** of data for each engine.
*   If an engine in the test set hasn't run long enough to fill a full window (e.g., it only has 20 cycles but the window is 30), that engine is dropped from the evaluation.

### 7. Multi-Dataset Execution
*   Finally, the code loops through four different datasets (**FD001 to FD004**). 
*   It applies a specific window size to each (e.g., 30 for FD001, 15 for FD004) because the operating conditions and failure modes differ across the sets.

**Summary of Output Shapes:**
After this cell runs, your data looks like this:
*   **X_train:** `(Samples, Window_Size, 14)`, The sensor inputs.
*   **y_train:** `(Samples,)`, The RUL values the model tries to learn.


### `model.py`
The `model.py` cell defines the "brain" of the system-a **Deep Convolutional Neural Network (DCNN)**. Its job is to look at the patterns in the sensor data over time and predict a single number: the Remaining Useful Life (RUL).

Here is the step-by-step breakdown of how the architecture works:

### 1. The Input Shape
The model expects a "window" of data. For example, if the window size is 30, it receives a 30x14 matrix (30 time steps of 14 different sensors). The code adds a "channel" dimension (like the colors in a photo), turning it into a **1x30x14** image for the computer to process.

### 2. Feature Extraction (The Convolutional Layers)
The model uses five layers of "filters" to scan the data:
*   **Layers 1 to 4:** These use 10 different filters each. The filters are sized **10x1**, meaning they look at 10 consecutive time steps for a **single sensor** at a time. They are looking for specific temporal shapes-like a sudden spike in temperature or a gradual drift in pressure.
*   **Layer 5:** This uses a smaller **3x1** filter to "summarize" the features found by the previous layers into a single consolidated map.
*   **Activation (`Tanh`):** After each layer, the model uses a `Tanh` function. This squashes the numbers and allows the model to learn complex, non-linear relationships in the engine data.

### 3. Regularization (`Dropout`)
Before making a final guess, the model uses a **Dropout** layer (set to 50%). During training, it randomly "turns off" half of the neurons. This forces the model to not rely too heavily on any single sensor or pattern, making it more robust and preventing it from simply memorizing the training data (overfitting).

### 4. The Regression Tail (Making the Prediction)
*   **Flattening:** The model takes the 2D "image" of features it has created and flattens it into one long list of numbers.
*   **Hidden Layer (`fc1`):** It passes these numbers into a layer of **100 neurons**. This layer combines all the high-level features extracted by the convolutions to understand the overall state of the engine.
*   **Output Layer (`fc2`):** Finally, it funnels everything into a single output neuron. This neuron outputs the final prediction: the number of cycles left until the engine fails.

### 5. Smart Initialization (`Xavier Normal`)
The code includes a special initialization function. Instead of starting the model's weights at random, it uses the **Xavier Normal** method. This ensures that the signals flowing through the network don't become too small or too large at the start of training, which helps the model learn much faster and more stably.

### Why this specific design?
This model is specifically a **2D CNN** applied to **1D time-series data**. By using a `(10, 1)` kernel, it treats each sensor independently in the first stage. It's essentially looking for "wear-and-tear" signatures within each sensor's timeline before combining those insights to make the final prediction.


### `train.py`
The `train.py` cell is the "practice session" where the model looks at the data, makes a guess, gets corrected, and repeats the process until it becomes accurate.

Here is the step-by-step breakdown:

### 1. The Penalty Systems (Loss and Scoring)
The code sets up two ways to judge the model:
*   **RMSE (Root Mean Squared Error):** This is the standard "mathematical" penalty. It calculates the average distance between the model's guess and the actual RUL. The model's goal during training is to get this number as low as possible.
*   **C-MAPSS Score:** This is a "real-world" penalty. In aviation, it is much worse to predict an engine has **more** life than it actually does (late prediction) than to predict it has **less** (early prediction). This score penalizes "late" guesses very heavily.

### 2. Organizing the Data
Before starting, the code converts the data into **Tensors** (the format PyTorch uses) and sends them to your **GPU** (if available) to speed up calculations. It also uses a **DataLoader** to feed the data to the model in chunks of **512 samples** at a time.

### 3. The Optimizer and "Smart Slowing" (Scheduler)
*   **Optimizer (Adam):** This is the algorithm that adjusts the model's internal weights. It starts with a "learning rate" of **0.001**.
*   **Scheduler:** After **200 rounds** (epochs) of training, the code automatically slows down the learning rate by 10x (to 0.0001). This is like a golfer slowing down their swing as they get closer to the hole-it helps the model "settle into" the best possible accuracy without overshooting it.

### 4. The Training Loop (250 Rounds)
The model goes through 250 full rounds of the training data. In each round:
1.  **Forward Pass:** The model looks at the 512 engine windows and makes guesses.
2.  **Calculate Error:** The `RMSELoss` function measures how wrong the guesses were.
3.  **Backpropagation:** The model works backward to figure out which "neurons" were responsible for the mistake.
4.  **Update:** The optimizer tweaks the weights to try and fix those mistakes for the next round.

### 5. Final Testing and Scoring
Once the 250 rounds are finished, the model is put into **Evaluation Mode**.
*   It is shown the **Test Set** (data it has never seen before).
*   It makes its final predictions for the engines in the test set.
*   The code calculates the final **RMSE** and the **C-MAPSS Score** to see how well the model actually learned.

### 6. Saving the Results
Finally, the code saves the model's "knowledge" into a file (e.g., `dcnn_model_FD001.pth`). This is crucial because it allows you to use the model later for evaluation or a report without having to spend the time and electricity to train it again from scratch.


### `evaluate.py`
The `evaluate.py` cell is the "Report Card" for your model. It takes the saved results from the training phase and creates a visual graph to show you exactly where the model is accurate and where it is making mistakes. Crucially, it is standalone: it loads the `.pth` files from disk, meaning you can run it without running the long training process first.

Here is the step-by-step breakdown:

### 1. Setting the Model to "Exam Mode"
The code calls `model.eval()`. This tells the model to stop using "Dropout" (the random turning off of neurons) and to act as a finished product. It also uses `torch.no_grad()`, which tells the computer: "Don't bother remembering the math for learning; we are only here to get answers."

### 2. Generating Predictions
The code takes the **Test Set** (the sensor data from engines that haven't failed yet) and runs it through the model. The model outputs its best guess for the Remaining Useful Life (RUL) for every single engine in the test set.

### 3. Sorting for Clarity
If the code just plotted the results randomly, the graph would look like a mess of dots. To fix this, the code **sorts the data**.
*   It organizes the engines so that the one with the most life remaining is on the left.
*   The engine closest to failure is on the right.
*   This creates a clean "descending staircase" shape for the ground truth.

### 4. Creating the Visualization
The code uses a library called `matplotlib` to draw a chart for each dataset (FD001, FD002, etc.):
*   **The Blue Line/Dots (Actual RUL):** This represents the "Ground Truth." It shows the real number of cycles left for each engine. Because we sorted it, it looks like a smooth line dropping from 125 down to zero.
*   **The Red Line/Stars (Predicted RUL):** These are the model's guesses. Each star corresponds to an engine. 

### 5. Interpreting the Results
By looking at this graph, you can quickly diagnose your model's health:
*   **Stars on the line:** The model is perfectly accurate.
*   **Stars above the line:** The model is being "too optimistic" (dangerous, as it thinks the engine will last longer than it will).
*   **Stars below the line:** The model is being "too cautious" (better for safety, but may lead to unnecessary maintenance).

### 6. Final Summary
The code concludes by embedding the dataset ID and final **RMSE** (average error in cycles) directly into the title of each plot, and printing out the **C-MAPSS Score** one last time for each dataset, giving you the final numbers you need for your project report.