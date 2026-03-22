import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class client_SisFall_1DCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(client_SisFall_1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # --- LAYER 1: The "Spike" Detector ---
            # Looks at raw 3D accelerometer data. 
            nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # --- LAYER 2: The "Context" Detector ---
            # Looks at the patterns found by Layer 1 to understand the sequence.
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Global Average Pooling to crush the timeline down to 1
            nn.AdaptiveAvgPool1d(1)
        )
        
        # --- CLASSIFIER ---
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class pretrain_SisFall_1DCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=19):
        super(pretrain_SisFall_1DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # --- LAYER 1: The "Spike" Detector ---
            # Looks at raw 3D accelerometer data. 
            nn.Conv1d(in_channels=input_channels, out_channels=8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # --- LAYER 2: The "Context" Detector ---
            # Looks at the patterns found by Layer 1 to understand the sequence.
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Global Average Pooling to crush the timeline down to 1
            nn.AdaptiveAvgPool1d(1)
        )
        
        # --- CLASSIFIER ---
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def pretrain_train_model(model, dataloader, num_epochs=5):
    """
    Trains the 1DCNN model for the pretrain clients.
    Trains on labels on the ADL for that trial on subjects without fall data.

    Returns: Model wieghts
    """

    # Define the Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Define the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- THE TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train() # Put model in 'training mode'
        
        running_loss = 0.0 
        
        # Iterate over batches in the dataloader
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}")
        
    print("\n[PRETRAINING] Pretraining complete. Returning global weights.")
    return model.state_dict()

def client_train_model(model, dataloader, num_epochs=5, use_DP = False):
    """
    Trains the 1DCNN model designed for clients with Differential Privacy for individual privacy.
    """

    # Intialize Privacy Engine
    if use_DP:
        privacy_engine = PrivacyEngine()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Attach the Privacy Engine using the TARGET EPSILON
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=num_epochs,        
            target_epsilon=10.0,     
            target_delta=1e-5, 
            max_grad_norm=8.0         
        )
    else:
        optimizer =  optim.Adam(model.parameters(), lr=0.001)
    
    
    class_weights = torch.tensor([1.0, 3.0]) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- THE TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train() # Put model in 'training mode'
        
        # Iterate over batches in the dataloader
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward Pass
            outputs = model(inputs)
            
            # 3. Calculate Loss
            loss = criterion(outputs, labels)
            
            # 4. Backward Pass (Backpropagation)
            loss.backward()
            
            # 5. Update the weights based on the gradients
            optimizer.step()

    return model

def test_model(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    # Lists to hold all labels and predictions for scikit-learn
    all_labels = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            
            # Standardize loss calculation
            loss += criterion(outputs, labels).item() * inputs.size(0)
            
            # Get the predicted class (0 for ADL, 1 for Fall)
            preds = outputs.argmax(1)
            
            # Standard accuracy components
            total += labels.size(0)
            correct += (preds == labels).type(torch.float).sum().item()
            
            # Move tensors to CPU, convert to numpy, and store them
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate overall averages
    avg_loss = loss / total
    accuracy = correct / total
    
    # Calculate advanced metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1