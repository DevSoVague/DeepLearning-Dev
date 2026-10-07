import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import LSTMModel, create_sequences

def train_model(dataset_id='FD001', window_size=50, batch_size=64, hidden_size=128, num_layers=2, learning_rate=0.001, epochs=100):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {dataset_id}")

    # 2. Load and Sequence Data
    file_path = f'train_preprocessed_{dataset_id}.csv'
    print(f"Loading preprocessed training data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found. Did you run preprocessing.py --dataset {dataset_id}?")
        return

    train_df = pd.read_csv(file_path)
    
    X_train, y_train = create_sequences(train_df, window_size)
    
    # Convert to PyTorch Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model, Loss, and Optimizer
    input_size = X_train.shape[2] # Dynamic input size based on available sensors
    print(f"Input size: {input_size} features")
    
    model = LSTMModel(input_size, hidden_size, num_layers).to(DEVICE)
    
    # MSELoss is the standard for numerical prediction tasks like RUL
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    history = []
    print(f"\nStarting Training ({epochs} epochs)...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # 5. Save the trained model
    model_name = f'lstm_model_{dataset_id}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'window_size': window_size
    }, model_name)
    print(f"\nModel saved as '{model_name}'")
    
    # 6. Plot the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title(f'Training Loss ({dataset_id})')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss')
    plt.grid(True)
    plot_name = f'training_loss_{dataset_id}.png'
    plt.savefig(plot_name)
    print(f"Loss plot saved as '{plot_name}'")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description='Train LSTM on CMAPSS data')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset ID (FD001-FD004)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--window', type=int, default=50, help='Window size')
    
    args = parser.parse_args()
    
    train_model(
        dataset_id=args.dataset,
        epochs=args.epochs,
        hidden_size=args.hidden,
        num_layers=args.layers,
        learning_rate=args.lr,
        batch_size=args.batch,
        window_size=args.window
    )

