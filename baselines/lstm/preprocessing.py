import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

# Define the data path (C-MAPSS text files in CMAPSSData/ at the repo root)
DATA_DIR = '../../CMAPSSData'
data_path = DATA_DIR

col_names = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

def load_data(file_name):
    file_path = os.path.join(data_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    return df

def calculate_train_rul(df):
    # Get max cycle for each unit
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    # Merge and calculate RUL
    df = df.merge(max_cycle, on=['unit'], how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def calculate_test_rul(test_df, rul_truth_file):
    # Load ground truth
    rul_truth = pd.read_csv(os.path.join(data_path, rul_truth_file), sep='\s+', header=None, names=['true_rul'])
    rul_truth['unit'] = rul_truth.index + 1
    # Get max cycle for test units
    test_max_cycle = test_df.groupby('unit')['cycle'].max().reset_index()
    test_max_cycle.columns = ['unit', 'max_cycle']
    # Merge and calculate RUL
    test_df = test_df.merge(test_max_cycle, on=['unit'], how='left')
    test_df = test_df.merge(rul_truth, on=['unit'], how='left')
    test_df['RUL'] = test_df['true_rul'] + (test_df['max_cycle'] - test_df['cycle'])
    test_df.drop(['max_cycle', 'true_rul'], axis=1, inplace=True)
    return test_df

def run_preprocessing(dataset_id='FD001'):
    print(f"\n--- Starting Preprocessing for {dataset_id} ---")
    
    # 1. Load Data
    print(f"Loading raw data for {dataset_id}...")
    train_df = load_data(f'train_{dataset_id}.txt')
    test_df = load_data(f'test_{dataset_id}.txt')

    # 2. Calculate RUL
    print("Calculating RUL...")
    train_df = calculate_train_rul(train_df)
    test_df = calculate_test_rul(test_df, f'RUL_{dataset_id}.txt')

    # 3. Clip RUL (Piecewise Linear RUL)
    CLIP_LIMIT = 125
    print(f"Clipping RUL at {CLIP_LIMIT}...")
    train_df['RUL'] = train_df['RUL'].clip(upper=CLIP_LIMIT)
    test_df['RUL'] = test_df['RUL'].clip(upper=CLIP_LIMIT)

    # 4. Identify and Drop Constant Sensors (Dynamic per dataset)
    print("Identifying constant sensors...")
    constant_sensors = []
    # Check all columns starting with 's' or 'os' (though settings usually vary)
    potential_cols = [col for col in train_df.columns if col.startswith('s') or col.startswith('os')]
    for col in potential_cols:
        if train_df[col].min() == train_df[col].max():
            constant_sensors.append(col)
    
    print(f"Dropping constant columns for {dataset_id}: {constant_sensors}")
    train_df.drop(columns=constant_sensors, inplace=True)
    test_df.drop(columns=constant_sensors, inplace=True)

    # 5. Normalization (Scaling to 0-1)
    # We normalize everything except the 'unit' ID and the 'RUL' answer.
    cols_to_normalize = train_df.columns.difference(['unit', 'RUL'])
    
    print(f"Normalizing {len(cols_to_normalize)} columns...")
    scaler = MinMaxScaler()
    
    # CRITICAL: We fit the scaler ONLY on the training data
    scaler.fit(train_df[cols_to_normalize])
    
    # Then we transform both train and test
    train_df[cols_to_normalize] = scaler.transform(train_df[cols_to_normalize])
    test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])

    # 6. Save the results
    train_out = f'train_preprocessed_{dataset_id}.csv'
    test_out = f'test_preprocessed_{dataset_id}.csv'
    print(f"Saving preprocessed data to {train_out} and {test_out}...")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    print(f"SUCCESS! {dataset_id} is ready. Final column count: {len(train_df.columns)}")

if __name__ == "__main__":
    # Support command line execution
    parser = argparse.ArgumentParser(description='Preprocess CMAPSS data')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset ID (FD001, FD002, FD003, FD004)')
    args = parser.parse_args()
    
    run_preprocessing(args.dataset)

