import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
import os
from model import LSTMModel

def evaluate_model(dataset_id='FD001'):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Dataset: {dataset_id}")

    # 2. Load the trained model checkpoint
    model_path = f'lstm_model_{dataset_id}.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Extract configuration from checkpoint
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    window_size = checkpoint['window_size']
    
    print(f"Loading model with input_size={input_size}, hidden_size={hidden_size}, window_size={window_size}")
    
    model = LSTMModel(input_size, hidden_size, num_layers).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Load the test data
    data_path = f'test_preprocessed_{dataset_id}.csv'
    print(f"Loading preprocessed test data from {data_path}...")
    test_df = pd.read_csv(data_path)
    
    # 4. Prepare sequences for the Test Set
    X_test = []
    y_test = []
    
    for unit in test_df['unit'].unique():
        unit_data = test_df[test_df['unit'] == unit]
        
        # If the engine has enough cycles, take the last window
        if len(unit_data) >= window_size:
            features = unit_data.drop(columns=['unit', 'RUL']).values
            window = features[-window_size:]
            label = unit_data['RUL'].iloc[-1]
            
            X_test.append(window)
            y_test.append(label)
        else:
            # Handle short sequences by zero-padding or using what we have
            features = unit_data.drop(columns=['unit', 'RUL']).values
            padding = np.zeros((window_size - len(unit_data), input_size))
            window = np.vstack((padding, features))
            label = unit_data['RUL'].iloc[-1]
            
            X_test.append(window)
            y_test.append(label)
            print(f"Note: Unit {unit} padded (only {len(unit_data)} cycles).")

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).to(DEVICE)
    y_test = np.array(y_test)
    
    # 5. Make Predictions
    print("Making predictions...")
    with torch.no_grad():
        predictions = model(X_test)
        predictions = predictions.cpu().numpy().flatten()
    
    # 6. Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\nFinal Test RMSE ({dataset_id}): {rmse:.2f}")
    
    # 7. Visualize Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual RUL', color='blue', marker='o', linestyle='', alpha=0.6)
    plt.plot(predictions, label='Predicted RUL', color='red', marker='x', linestyle='', alpha=0.8)
    plt.title(f'Actual vs Predicted RUL - {dataset_id} (Test Set)')
    plt.xlabel('Engine ID (Index)')
    plt.ylabel('Remaining Useful Life (Cycles)')
    plt.legend()
    plt.grid(True)
    
    plot_name = f'evaluation_results_{dataset_id}.png'
    plt.savefig(plot_name)
    print(f"Results plot saved as '{plot_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LSTM on CMAPSS data')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset ID (FD001-FD004)')
    args = parser.parse_args()
    
    evaluate_model(args.dataset)

