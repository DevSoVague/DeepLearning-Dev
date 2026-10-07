import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    """
    The 'Brain' of our project. 
    It takes a sequence of sensor data and predicts the Remaining Useful Life.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        
        # 1. The LSTM Layer: This is where the 'memory' lives.
        # batch_first=True means our data looks like (Batch, Time, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 2. A Dropout layer: This randomly turns off some neurons during training.
        self.dropout = nn.Dropout(0.2)
        
        # 3. The Fully Connected Layer: Turns the LSTM's memory into a single prediction.
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x is our window of data (e.g., 50 cycles of 19 sensors)
        
        # Pass through LSTM
        # out contains the 'hidden state' for every step in the sequence
        out, _ = self.lstm(x)
        
        # We only care about the very LAST step in the sequence 
        # (the most recent health state of the engine)
        out = out[:, -1, :]
        
        # Apply dropout and the final linear layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

def create_sequences(data, window_size):
    """
    Helper function to turn flat data into sequences/windows.
    """
    sequences = []
    labels = []
    
    # We process each engine (unit) separately
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit]
        
        # Feature columns (everything except unit and RUL)
        features = unit_data.drop(columns=['unit', 'RUL']).values
        target = unit_data['RUL'].values
        
        # Slide the window across the engine's life
        for i in range(len(unit_data) - window_size + 1):
            window = features[i : i + window_size]
            label = target[i + window_size - 1]
            
            sequences.append(window)
            labels.append(label)
            
    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    print("Model definition script. Use train.py to actually train this model.")
