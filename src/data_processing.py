import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

def complete_dataload(root_path = r"..\data\SisFall_dataset"):
    """
    Orchestrates the entire loading and processing pipeline.
    Returns:
        pretrain_loader: DataLoader for centralized pre-training.
        fl_loaders: Dict { 'SA06': DataLoader, ... } for Federated Learning.
    """
    
    # Client Ids to put in pretrain set
    pretrain_clients = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05']
    
    # Load Raw Data
    print("Step 1: Loading raw SisFall data into memory...")
    raw_data_dict = load_sisfall_for_fl(root_path)
    
    pretrain_trials = []
    fl_loaders = {}

    print("Step 2: Processing clients...")
    
    # Iterate through every subject found
    for client_id, trials in raw_data_dict.items():
        
        # Pre-train vs Federated Client
        if client_id in pretrain_clients:
            # Collect all trials for later centralized processing
            pretrain_trials.extend(trials)
        else:
            # Process this client immediately for FL
            X_client, y_client = process_trials_individually(trials)
            
            # Only create loader if valid data exists
            if len(X_client) > 0:
                fl_loaders[client_id] = create_dataloader(X_client, y_client)
                
    # Process the collected Pre-train Data
    print(f"Step 3: Finalizing Pre-train set ({len(pretrain_clients)} clients)...")
    X_pre, y_pre = process_trials_individually(pretrain_trials)
    pretrain_loader = create_dataloader(X_pre, y_pre)

    print(f"\n=== READY ===")
    print(f"Pre-train Set: {len(pretrain_loader.dataset)} windows")
    print(f"FL Clients: {len(fl_loaders)} active clients")
    
    return pretrain_loader, fl_loaders

def load_sisfall_for_fl(root_path, select_sensors='acc_only'):
    """
    Loads SisFall data into a client-partitioned dictionary.
    
    Args:
        root_path: Path to the unzipped SisFall folder.
        select_sensors: 'all' or 'acc_only' to save memory.
        
    Returns:
        clients_data: Dict { 'client_id': df }
        df: all the sensor data for that subject and trial
    """

    col_names = ["ADXL_x", "ADXL_y", "ADXL_z"]

    clients_data = {}

    # Walk through the directory
    # Expected structure: SisFall_dataset/SA01/D01_SA01_R01.txt
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                
                # 1. Parse Metadata from filename
                # Filename format typically: Activity_Subject_Trial.txt (e.g., D01_SA01_R01.txt)
                try:
                    parts = file.replace('.txt', '').split('_')
                    activity_code = parts[0] # e.g., D01
                    subject_id = parts[1]    # e.g., SA01 (This is our Client)
                except IndexError:
                    continue # Skip readme or unexpected files

                # 2. Load the file
                file_path = os.path.join(root, file)
                
                # Read CSV and only use the first three columns
                df_temp = pd.read_csv(file_path, header=None, 
                                      names = col_names, usecols=[0, 1, 2], dtype='float32')
                
                # 3. Labeling fall binary value (1 if fall, 0 ADLs)
                is_fall = 1 if activity_code.startswith('F') else 0
                df_temp['label'] = is_fall
                # Adding activity tag for finding fall for sliding windows
                df_temp['activity'] = activity_code
                
                # 4. Append to Client's bucket
                if subject_id not in clients_data:
                    clients_data[subject_id] = []
                
                clients_data[subject_id].append(df_temp)

    return clients_data

def process_trials_individually(trials_list, window_size=128):
    """
    Optimized for IoT: Downsamples to 50Hz and uses smaller windows.
    """
    # Find radius for window center at SVM spike
    radius = window_size // 2
    
    processed_windows = []
    labels = []

    for df in trials_list:
        # === DOWNSAMPLE ===
        # Reduce down to 50 hz from 200 hz i.e take every 4th sample
        df_small = df.iloc[::4, :].reset_index(drop=True)
        
        act_code = df_small['activity'].iloc[0]
        sensor_data = df_small[['ADXL_x', 'ADXL_y', 'ADXL_z']].values

        # === FALL PROCESSING ===
        if act_code.startswith('F'):
            # Recalculate SVM on the DOWNSAMPLED data
            svm = np.sqrt(np.sum(sensor_data**2, axis=1))
            # Find the max SVP from data
            impact_idx = np.argmax(svm)

            # Shift by approx +/- 0.05 seconds (Time Jitter)
            # 0.05s * 50Hz = 2.5 samples -> round to 3
            jitter = 3
            shifts = [-jitter, 0, jitter] 

            for shift in shifts:
                # Shift the center using jitter
                center = impact_idx + shift
                # Bounds check
                if center - radius >= 0 and center + radius < len(df_small):
                    window = sensor_data[center - radius : center + radius]
                    processed_windows.append(window)
                    labels.append(1)

        # === ADL PROCESSING (Pooled Random) ===
        else:
            # Stride of half the window size (about 1.3 seconds); allows for 50% overlap
            stride = window_size // 2
            for i in range(0, len(df_small) - window_size, stride):
                window = sensor_data[i : i + window_size]
                # We defer selection until later (Pooled Random Strategy)
                processed_windows.append(window)
                labels.append(0)

    # Convert to Arrays
    X_all = np.array(processed_windows)
    y_all = np.array(labels)
    
    # === STEP 4: BALANCE THE DATA ===
    # Separate Falls and ADLs
    falls_idx = np.where(y_all == 1)[0]
    adls_idx = np.where(y_all == 0)[0]
    
    # Target 3x ADLs
    target_adls = len(falls_idx) * 3
    
    if len(adls_idx) > target_adls:
        selected_adls = np.random.choice(adls_idx, target_adls, replace=False)
    else:
        selected_adls = adls_idx
        
    final_indices = np.concatenate([falls_idx, selected_adls])
    np.random.shuffle(final_indices)
    
    return X_all[final_indices], y_all[final_indices]

def create_dataloader(X, y):
    """
    Wraps numpy arrays into PyTorch DataLoader.
    """
    class SisFallDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): 
            # Transpose [Time, Channel] -> [Channel, Time] for CNN
            return self.X[idx].transpose(0, 1), self.y[idx]

    dataset = SisFallDataset(X, y)
    return DataLoader(dataset, batch_size = 32, shuffle=True)