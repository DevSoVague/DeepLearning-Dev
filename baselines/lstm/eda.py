import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the data path (C-MAPSS text files in CMAPSSData/ at the repo root)
DATA_DIR = '../../CMAPSSData'
data_path = DATA_DIR

# Column names based on the readme.txt
# 1) unit number
# 2) time, in cycles
# 3) operational setting 1
# 4) operational setting 2
# 5) operational setting 3
# 6-26) sensor measurement 1-21

col_names = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

def load_data(file_name):
    file_path = os.path.join(data_path, file_name)
    # The data is space separated, and there are trailing spaces in the files
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    return df

if __name__ == "__main__":
    print("Loading FD001 training data...")
    train_df = load_data('train_FD001.txt')
    
    print("\nData Shape:", train_df.shape)
    print("\nFirst 5 rows:")
    print(train_df.head())
    
    print("\nData Info:")
    print(train_df.info())
    
    # Calculate Remaining Useful Life (RUL) for each row
    # RUL = Max Cycle for that unit - Current Cycle
    print("\nCalculating RUL...")
    
    # Get the maximum cycle for each unit
    max_cycle = train_df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    
    # Merge back to the original dataframe
    train_df = train_df.merge(max_cycle, on=['unit'], how='left')
    
    # Calculate RUL
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    
    # Drop max_cycle as it's no longer needed
    train_df.drop('max_cycle', axis=1, inplace=True)
    
    print("\nFirst 5 rows with RUL:")
    print(train_df[['unit', 'cycle', 'RUL']].head())
    
    # Plotting sensor 2 for unit 1
    plt.figure(figsize=(10, 5))
    unit1 = train_df[train_df['unit'] == 1]
    plt.plot(unit1['cycle'], unit1['s2'], label='Sensor 2')
    plt.xlabel('Cycle')
    plt.ylabel('Sensor Value')
    plt.title('Sensor 2 trend for Unit 1')
    plt.legend()
    plt.grid(True)
    plt.savefig('sensor_trend_unit1.png')
    print("\nSaved sensor trend plot to 'sensor_trend_unit1.png'")

    # --- Process Test Data ---
    print("\nLoading FD001 test data and ground truth RUL...")
    test_df = load_data('test_FD001.txt')
    
    # Load the ground truth RUL for the test set
    rul_truth = pd.read_csv(os.path.join(data_path, 'RUL_FD001.txt'), sep='\s+', header=None, names=['true_rul'])
    rul_truth['unit'] = rul_truth.index + 1 # Engine IDs start at 1
    
    # Get the last cycle for each unit in the test set
    test_max_cycle = test_df.groupby('unit')['cycle'].max().reset_index()
    test_max_cycle.columns = ['unit', 'max_cycle']
    
    # Merge the true RUL with the max cycle info
    test_df = test_df.merge(test_max_cycle, on=['unit'], how='left')
    test_df = test_df.merge(rul_truth, on=['unit'], how='left')
    
    # Calculate RUL for every row in the test set
    # RUL = (True RUL at end) + (Max Cycle - Current Cycle)
    test_df['RUL'] = test_df['true_rul'] + (test_df['max_cycle'] - test_df['cycle'])
    
    # Clean up temporary columns
    test_df.drop(['max_cycle', 'true_rul'], axis=1, inplace=True)
    
    print("First 5 rows of Test Data with RUL:")
    print(test_df[['unit', 'cycle', 'RUL']].head())

    # Save both tables to CSV files
    train_df.to_csv('processed_train_data.csv', index=False)
    test_df.to_csv('processed_test_data.csv', index=False)
    print("\nSUCCESS: Both 'processed_train_data.csv' and 'processed_test_data.csv' have been saved!")
