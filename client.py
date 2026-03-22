import psutil
import sys
import os
import json
import flwr as fl
import torch
import numpy as np
import pickle
from lightphe import LightPHE
from collections import OrderedDict
from model import client_train_model, test_model, client_SisFall_1DCNN
from data_processing import individual_dataload

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader, use_encryption=False, use_DiffPrivacy = False):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.use_encryption = use_encryption
        self.use_DiffPrivacy = use_DiffPrivacy
        
        self.tensor_shapes = [val.shape for _, val in self.model.state_dict().items()]
        
        # Read the JSON file and parse it into a Python dictionary
        with open("lightphe_keys.json", "r") as f:
            key_data = json.load(f)
            
        # Pass the dictionary into the LightPHE cryptosystem
        self.cs = LightPHE(algorithm_name="Paillier", keys=key_data)

    def get_parameters(self, config):
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        if self.use_encryption:
            # Flatten into a standard Python list of floats
            flat_weights = np.concatenate([w.flatten() for w in weights]).tolist()
            
            # Shift weights to be strictly positive
            SHIFT_OFFSET = 10.0
            shifted_weights = [w + SHIFT_OFFSET for w in flat_weights]
            
            # Encrypt the strictly positive weights
            ciphertext = self.cs.encrypt(shifted_weights)
            
            # Pickle the ciphertext object into bytes for Flower
            return [pickle.dumps(ciphertext)]
        else:
            return weights

    def set_parameters(self, parameters):
        if self.use_encryption:
            ciphertext = pickle.loads(parameters[0])
            decrypted_flat_weights = self.cs.decrypt(ciphertext)
            
            SHIFT_OFFSET = 10.0

            if len(parameters) > 1:
                num_clients = parameters[1][0] 
                # Reverse the aggregation division and subtract the shift
                decrypted_flat_weights = [(w / num_clients) - SHIFT_OFFSET for w in decrypted_flat_weights]
            else:
                # If no aggregation happened, just remove the shift
                decrypted_flat_weights = [w - SHIFT_OFFSET for w in decrypted_flat_weights]
            
            weights = []
            current_index = 0
            for shape in self.tensor_shapes:
                num_elements = np.prod(shape)
                
                layer_weights = np.array(
                    decrypted_flat_weights[current_index : current_index + num_elements],
                    dtype=np.float32
                ).reshape(shape)
                
                weights.append(layer_weights)
                current_index += num_elements
        else:
            weights = [np.array(w, dtype=np.float32) for w in parameters]

        params_dict = zip(self.model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        client_train_model(self.model, self.trainloader, num_epochs=3, use_DP = self.use_DiffPrivacy)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Unpack the new metrics
        loss, accuracy, precision, recall, f1 = test_model(self.model, self.testloader)
        
        # Capture System Resources
        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()
    
        # Capture Payload Size 
        payload_size_mb = sum(sys.getsizeof(t) for t in parameters) / (1024 * 1024)
    
        # Pack EVERYTHING into the metrics dictionary for the server
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "ram_usage_mb": ram_usage_mb,
            "cpu_usage_percent": cpu_usage,
            "payload_size_mb": payload_size_mb
        }

        return float(loss), len(self.testloader.dataset), metrics

def make_client_fn(subject_names, pretrained_weights,  use_encryption, use_DiffPrivacy):
    """
    Creates an individual client with its own data and model.
    
    Args:
        subject_names: The list of the subjects to create clients for (e.g., ['SA01','SA02']
        pretrained_weights: The weights resulting from the pretrained model
        use_encryption: Boolean value, whether the clients use HE for aggregation
        
    Returns:
        client_fn = client designed for Flower simulation
    """
    
    # Flower will only call this inner function
    def client_fn(cid: str) -> fl.client.Client:
        # It can "see" subject_names and use_encryption from the outer scope!
        real_subject_id = subject_names[int(cid)]
        
        trainloader, testloader = individual_dataload(real_subject_id)
        model = client_SisFall_1DCNN()
        
        #  Delete Final layer of Pretrained Weights
        pretrained_dict = pretrained_weights.copy()
    
        keys_to_delete = []
        for key in pretrained_dict.keys():
            if 'classifier.' in key:  
                keys_to_delete.append(key)
            
        for key in keys_to_delete:
            del pretrained_dict[key]
        
        # Load the filtered weights 
        model.load_state_dict(pretrained_dict, strict=False)
        
        return FlowerClient(
            cid=real_subject_id,
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            use_encryption=use_encryption,
            use_DiffPrivacy= use_DiffPrivacy
        ).to_client()
        
    # Return the customized inner function back to the notebook
    return client_fn