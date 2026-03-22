import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Create this globally so all clients use the exact same mapping
SISFALL_LABEL_MAP = {
    0: 0,   # Binary ADL label
    1: 1,   # Binary Fall Label
    'D01': 2, 'D02': 3, 'D03': 4, 'D04': 5, 'D05': 6,
    'D06': 7, 'D07': 8, 'D08': 9, 'D09': 10, 'D10': 11,
    'D11': 12, 'D12': 13, 'D13': 14, 'D14': 15, 'D15': 16,
    'D16': 17, 'D17': 18, 'D18': 19, 'D19': 20
}

ROOT_PATH = r"data\SisFall_dataset"

def complete_dataload(root_path = ROOT_PATH):
    """
    Orchestrates the entire loading and processing pipeline.
    Returns:
        pretrain_loader: DataLoader for centralized pre-training.
        fl_loaders: Dict { 'SA06': DataLoader, ... } for Federated Learning.
    """
    
    # Client Ids to put in pretrain set
    pretrain_clients = ["SE01"]
    
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
    print(f"Pre-train Set: {len(pretrain_loader[0].dataset)} windows")
    print(f"FL Clients: {len(fl_loaders)} active clients")
    
    return pretrain_loader, fl_loaders


def individual_dataload(subject_id, root_path = ROOT_PATH):
    """
    Loads raw data files for a single SisFall subject into DataLoaders.
    
    Args:
        subject_id: The string ID of the subject to load (e.g., 'SA01')
        root_path: Path to the main SisFall dataset directory 
        
    Returns:
        (trainloader, test_loader)
    """
    
    # Construct the path to the specific subject's folder
    # Expected: ./SisFall_dataset/SA01
    subject_folder = os.path.join(root_path, subject_id)
    
    if not os.path.exists(subject_folder):
        raise FileNotFoundError(f"Could not find folder for subject {subject_id} at {subject_folder}")

    cols = ["ADXL_x", "ADXL_y", "ADXL_z"]

    trials_list = []

    # Iterate ONLY through the files in this specific subject's folder
    for file in os.listdir(subject_folder):
        if file.endswith(".txt"):
            
            # 1. Parse Metadata
            # Filename format: Activity_Subject_Trial.txt (e.g., D01_SA01_R01.txt)
            parts = file.replace('.txt', '').split('_')
                
            activity_code = parts[0] # e.g., D01 or F04
            
            # 2. Load the file
            file_path = os.path.join(subject_folder, file)
            df_temp = pd.read_csv(file_path, header=None, 
                                      names = cols, usecols=[0, 1, 2], dtype='float32')
            
            # 3. Apply Labels
            df_temp['activity'] = activity_code
            df_temp['label'] = 1 if activity_code.startswith('F') else 0
            
            # 4. Append to our list
            trials_list.append(df_temp)
            
    X_client, y_client = process_trials_individually(trials_list)
    client_dataloader = create_dataloader(X_client, y_client)
            
    return client_dataloader

def pretrain_dataload(subject_list, root_path = ROOT_PATH):
    """
    Loads raw data files of SisFall subject into DataLoaders for pretraining.
    
    Args:
        subject_list: The list of the subjects to load for pretraining (e.g., 'SE01')
        root_path: Path to the main SisFall dataset directory 
        
    Returns:
        pretrain_dataloader
    """
    folder_list = []
    for subject in subject_list:
        subject_folder = os.path.join(root_path, subject)
        if not os.path.exists(subject_folder):
            raise FileNotFoundError(f"Could not find folder for subject {subject}")
        else:
            folder_list.append(subject_folder)

    cols = ["ADXL_x", "ADXL_y", "ADXL_z"]

    trials_list = []

    # Iterate through the files in pretrain list
    for subject_folder in folder_list:
        for file in os.listdir(subject_folder):
            if file.endswith(".txt"):
                
                # 1. Parse Metadata
                # Filename format: Activity_Subject_Trial.txt (e.g., D01_SA01_R01.txt)
                parts = file.replace('.txt', '').split('_')
                    
                activity_code = parts[0] # e.g., D01 or F04
                
                # 2. Load the file
                file_path = os.path.join(subject_folder, file)
                df_temp = pd.read_csv(file_path, header=None, 
                                        names = cols, usecols=[0, 1, 2], dtype='float32')
                
                # 3. Apply Labels
                df_temp['activity'] = activity_code
                df_temp['label'] = 1 if activity_code.startswith('F') else 0
                
                # 4. Append to our list
                trials_list.append(df_temp)
            
    X_client, y_client = process_trials_individually(trials_list)
    client_dataloader = create_dataloader(X_client, y_client)
            
    return client_dataloader


def load_sisfall_for_fl(root_path):
    """
    Loads SisFall data into a client-partitioned dictionary.
    
    Args:
        root_path: Path to the unzipped SisFall folder.
        
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

def process_trials_individually(trials_list, window_size=128, no_fall_cap=150):
    """
    Downsamples to 50Hz and uses smaller windows.
    Subjects with falls: Binary labels (1 = Fall, 0 = ADL).
    Subjects without falls: Multi-class labels (e.g., 'D01', 'D02').
    """
    radius = window_size // 2
    processed_windows = []
    labels = []

    # === CHECK FOR FALLS ===
    has_falls = any(df['activity'].iloc[0].startswith('F') for df in trials_list)

    for df in trials_list:
        # === DOWNSAMPLE ===
        df_small = df.iloc[::4, :].reset_index(drop=True)
        
        act_code = df_small['activity'].iloc[0]
        sensor_data = df_small[['ADXL_x', 'ADXL_y', 'ADXL_z']].values

        # === FALL PROCESSING ===
        if act_code.startswith('F'):
            svm = np.sqrt(np.sum(sensor_data**2, axis=1))
            impact_idx = np.argmax(svm)
            jitter = 3
            shifts = [-jitter, 0, jitter] 

            for shift in shifts:
                center = impact_idx + shift
                if center - radius >= 0 and center + radius < len(df_small):
                    window = sensor_data[center - radius : center + radius]
                    processed_windows.append(window)
                    labels.append(1)

        # === ADL PROCESSING ===
        else:
            stride = window_size // 2
            for i in range(0, len(df_small) - window_size, stride):
                window = sensor_data[i : i + window_size]
                processed_windows.append(window)
                
                # Dynamic Labeling based on your new design
                if has_falls:
                    labels.append(0) # Standard binary 
                else:
                    labels.append(act_code) # Specific ADL label for subjects without falls

    if len(processed_windows) == 0:
        return np.array([]), np.array([])

    # Convert to Arrays
    X_all = np.array(processed_windows)
    y_all = np.array(labels)
    
    # === BALANCE THE DATA ===
    if has_falls:
        # 1:3 balancing for subjects with falls
        falls_idx = np.where((y_all == 1))[0]
        adls_idx = np.where((y_all == 0))[0]
        
        target_adls = len(falls_idx) * 3
        
        if len(adls_idx) > target_adls:
            selected_adls = np.random.choice(adls_idx, target_adls, replace=False)
        else:
            selected_adls = adls_idx
            
        final_indices = np.concatenate([falls_idx, selected_adls])
        np.random.shuffle(final_indices)
        
        # Ensure binary labels are integers before returning
        return X_all[final_indices], y_all[final_indices].astype(int)
        
    else:
        # For sibjects with no fall data, randomly sample up to `no_fall_cap`
        adls_idx = np.arange(len(y_all))
        
        if len(adls_idx) > no_fall_cap:
            final_indices = np.random.choice(adls_idx, no_fall_cap, replace=False)
        else:
            final_indices = adls_idx
            
        np.random.shuffle(final_indices)
        
        # Returning string labels (e.g., 'D01')
        return X_all[final_indices], y_all[final_indices]
    
def create_dataloader(X, y):
    """
    Wraps numpy arrays into PyTorch DataLoader.
    Returns:
        train_DataLoader: DataLoader for training.
        test_DataLoader: DataLoader for testing.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    class SisFallDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            # Map strings/ints to their global class integers
            mapped_y = np.array([SISFALL_LABEL_MAP[label] for label in y])
            self.y = torch.from_numpy(mapped_y).long()

        def __len__(self): 
            return len(self.X)
        
        def __getitem__(self, idx): 
            # Transpose [Time, Channel] -> [Channel, Time] for CNN
            return self.X[idx].transpose(0, 1), self.y[idx]

    train_dataset = SisFallDataset(X_train, y_train)
    test_dataset = SisFallDataset(X_test, y_test)

    # Create the DataLoaders
    train_DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_DataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_DataLoader, test_DataLoader