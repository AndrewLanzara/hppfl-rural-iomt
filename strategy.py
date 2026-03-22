import pickle
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters

class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, use_encryption=False, *args, **kwargs):
        """
        Inherit from Flower's standard FedAvg strategy.
        We pass all standard args (like fraction_fit) to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.use_encryption = use_encryption

    # NOTE THE INDENTATION HERE! This must be inside the class.
    def aggregate_fit(self, server_round, results, failures):
        """Override the standard aggregation to handle LightPHE ciphertexts."""
        
        # If no clients succeeded, return empty
        if not results:
            return None, {}

        # Standard Plaintext Aggregation
        if not self.use_encryption:
            # Let Flower's built-in FedAvg handle the standard NumPy arrays
            return super().aggregate_fit(server_round, results, failures)

        # Homomorphic Aggregation
        print(f"\n--- Server Round {server_round}: Aggregating Encrypted Weights ---")
            
        # Calculate the total training examples across all participating clients
        total_examples = sum([fit_res.num_examples for _, fit_res in results])

        aggregated_ciphertext = None

        # Iterate through each client's encrypted weights
        for client, fit_res in results:
            # Unpack and unpickle
            unpacked_arrays = parameters_to_ndarrays(fit_res.parameters)
            raw_bytes = unpacked_arrays[0].tobytes() if unpacked_arrays[0].ndim > 0 else unpacked_arrays[0].item()
            client_ciphertext = pickle.loads(raw_bytes)
    
            # --- THE FIX: Pure Homomorphic Addition (No Multiplication!) ---
            if aggregated_ciphertext is None:
                aggregated_ciphertext = client_ciphertext
            else:
                aggregated_ciphertext = aggregated_ciphertext + client_ciphertext

        # Re-pickle the newly aggregated global ciphertext
        serialized_aggregated = pickle.dumps(aggregated_ciphertext)

        # Pass the number of clients (e.g., 2) to act as the denominator
        num_clients = len(results)
        aggregated_parameters = ndarrays_to_parameters([
            np.array(serialized_aggregated), 
            np.array([num_clients], dtype=np.float32) 
        ])

        return aggregated_parameters, {}

def weighted_average_metrics(eval_metrics):
    """
    Aggregates metrics returned by the clients during evaluation.
    eval_metrics is a list of tuples: (num_examples, client_metrics_dict)
    """
    if not eval_metrics:
        return {}

    # Extract the total number of examples across all clients
    total_examples = sum([num_examples for num_examples, _ in eval_metrics])
    num_clients = len(eval_metrics)

    # 1. Calculate Weighted Averages for ML Metrics
    weighted_accuracy = sum([num * m["accuracy"] for num, m in eval_metrics]) / total_examples
    weighted_f1 = sum([num * m["f1_score"] for num, m in eval_metrics]) / total_examples
    weighted_precision = sum([num * m["precision"] for num, m in eval_metrics]) / total_examples
    weighted_recall = sum([num * m["recall"] for num, m in eval_metrics]) / total_examples

    # 2. Calculate Simple Averages for System Resources
    avg_ram = sum([m["ram_usage_mb"] for _, m in eval_metrics]) / num_clients
    avg_cpu = sum([m["cpu_usage_percent"] for _, m in eval_metrics]) / num_clients
    avg_payload = sum([m["payload_size_mb"] for _, m in eval_metrics]) / num_clients

    # Return the dictionary that the Server will actually print and save!
    return {
        "accuracy": weighted_accuracy,
        "f1_score": weighted_f1,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "avg_ram_mb": avg_ram,
        "avg_cpu_percent": avg_cpu,
        "avg_payload_mb": avg_payload
    }