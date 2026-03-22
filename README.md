# Privacy-Preserving Federated Learning for Wearable Fall Detection

## Project Overview
This project implements a secure, edge-optimized machine learning framework for detecting falls using wearable sensor data. It utilizes a lightweight 1D-CNN (under 1,000 parameters) designed specifically to handle the heavy computational constraints of privacy-preserving technologies. 

To ensure user data never leaves the wearable device in a vulnerable state, this framework combines **Federated Learning (Flower)**, **Differential Privacy (Opacus)**, and **Homomorphic Encryption (Paillier)**. 

## Methodology & Design Choices

To balance the strict computational limits of Homomorphic Encryption with the clinical necessity of high-accuracy fall detection, several key experimental design choices were made:

### 1. Data Processing & Sensor Optimization
To ensure the model can run on low-power wearable microcontrollers, the raw SisFall dataset was heavily optimized for memory efficiency and battery life.
* **Action:** Sensor input was restricted to Accelerometer-only data (ADXL), and the sampling rate was downsampled from 200 Hz to 50 Hz.
* **Reasoning:** Gyroscopes consume significantly more power (roughly 3-5 mA compared to an accelerometer's 10-100 $\mu$ A), making them impractical for always-on wearables. Downsampling to 50 Hz satisfies the Nyquist-Shannon Theorem for human movement (which peaks around 20-25 Hz) while drastically compressing the input dimensionality.

### 2. Sliding Window Segmentation
While some literature suggests long capture windows (8-12 seconds) to capture "pre-fall" instability, this framework utilizes a highly compact 2.56-second window ($N=128$ samples).
* **Action:** ADL data is extracted using a 50% overlap stride, while Fall data uses peak-centered extraction.
* **Reasoning:** A smaller window minimizes the input tensor size, saving crucial RAM during the heavy Homomorphic Encryption aggregation phase. It also guarantees ultra-low latency inference, ensuring a fall is detected within seconds of impact rather than waiting for a massive buffer to fill.

### 3. Peak-Centered Extraction & Temporal Jitter
A 15-second "Fall" file contains mostly non-fall data (e.g., standing before the fall, lying still afterward). Feeding this entire file to the model introduces massive label noise and "Negative Transfer."
* **Action:** The Signal Vector Magnitude (SVM) is calculated using $\sqrt{x^2+y^2+z^2}$ to locate the exact impact spike. Windows are then center-cropped exactly on this impact moment, and the surrounding "silence" is discarded. Furthermore, to mimic real-world asynchronous streaming, **Temporal Jitter** was applied by augmenting shifts of $\pm 3$ samples around the impact center. 
* **Reasoning:** Discarding the silence prevents the model from mistakenly learning that standing still is a feature of falling. Applying temporal jitter (a ~60ms shift at 50 Hz) forces the 1D-CNN to become spatially invariant, ensuring it can still detect an impact even if the physical fall is slightly misaligned within the wearable's rolling inference window.

### 4. Differential Privacy Budget & Gradient Clipping
To mathematically guarantee that no individual subject's gait data can be reverse-engineered from the global model, the Opacus privacy engine was used to clip gradients and inject calibrated noise during local training.
* **Action:** The privacy budget was set to ε = 10.0 (δ = 10⁻⁵), and the maximum gradient clipping norm was strictly bounded at C = 8.0.
* **Reasoning:** A budget of ε = 10.0 provides strong plausible deniability while retaining enough signal-to-noise ratio to accurately identify critical fall events. The clipping norm of 8.0 was empirically determined by analyzing the unclipped local gradients of two sample subjects. Their gradient norms exhibited a highly skewed distribution, ranging from 0.95 to 39.7, with a median of 6.6. Selecting a clipping threshold just above the median (C = 8.0) ensures that extreme outlier movements do not disproportionately explode the DP noise multiplier, while cleanly preserving the natural learning signal for the vast majority of the data.

### 5. Client Sampling Rate
Rather than aggregating all 26 federated clients simultaneously, the simulation uses a `fraction_fit` of 0.2.
* **Action:** Only 3 to 4 clients are randomly selected to participate in each communication round.
* **Reasoning:** This accurately mimics the reality of wearable devices, where only a fraction of edge devices are charging and connected to Wi-Fi at any given time. It also prevents the central server from experiencing memory overflow during the heavy Homomorphic Encryption aggregation phase.

### 6. Transfer Learning via Public Pretraining
Because the dataset was aggressively capped to combat class imbalance, individual FL clients did not possess enough data to train a neural network from scratch under the heavy noise of Differential Privacy. 
* **Action:** A Transfer Learning approach was utilized. The global model was pretrained in plaintext using "public subjects" (excluding any sensitive fall data). This pretraining tasked the model with classifying 19 unique ADLs. The resulting feature-extractor was then distributed to the FL clients to be fine-tuned into a binary Fall/ADL classifier.
* **Reasoning:** Pretraining on 19 distinct ADLs forces the network to learn the fundamental physics and spatial hierarchies of human movement. When deployed to the FL clients, the model only needed to learn the specific features of an "impact," drastically reducing the amount of secure, on-device training required.

### 7. Ultra-Lightweight 1D-CNN Architecture
Homomorphic Encryption using the Paillier cryptosystem is computationally expensive. Performing encrypted additions on massive weight tensors would cause the server to time out.
* **Action:** The core classifier was designed as a custom, ultra-lightweight 1D Convolutional Neural Network containing roughly 500 total trainable parameters. 
* **Reasoning:** Standard time-series models (like LSTMs or deep ResNets) contain tens to hundreds of thousands of parameters. By utilizing a highly compressed 1D-CNN, the model successfully captures the 50 Hz kinematic data while keeping the "Homomorphic Encryption Tax" (aggregation time and encrypted payload size) feasible for real-world edge deployment.

### 8. Algorithmic Optimization for DP and Imbalance
To accommodate both the mathematical constraints of Differential Privacy and the clinical risk of missing a fall event, the local client training loops were customized.
* **Action:** A weighted Cross-Entropy Loss `[1.0, 3.0]` was applied to the binary classification. Additionally, the local optimizer dynamically switches depending on the privacy state: `Adam` is used for plaintext training, while `SGD` (Stochastic Gradient Descent) is used when the Opacus Privacy Engine is attached.
* **Reasoning:** The `[1.0, 3.0]` class weights heavily penalize False Negatives, directly resulting in the model's higher Recall rate (prioritizing elderly safety over false alarms). Furthermore, while Adam converges faster in plaintext, SGD is utilized during DP training because its linear gradient updates are far more stable and predictable when clipped and injected with Opacus noise.

##  Model Architecture & Hyperparameters

To ensure full reproducibility and to prove the viability of edge-device deployment under Homomorphic Encryption constraints, the exact network architecture and training hyperparameters are detailed below.

### 1D-CNN Architecture (562 Total Parameters)
The network is intentionally shallow to prevent the Paillier encryption aggregation time from exceeding practical limits. It utilizes Global Average Pooling (GAP) to collapse the temporal dimension, drastically reducing the parameters required for the final dense layer.

* **Input Layer:** 128 timesteps $\times$ 3 channels (Raw 3D Accelerometer data: ADXL_x, ADXL_y, ADXL_z).
* **Layer 1 (The "Spike" Detector):** * 1D Convolution (8 filters, kernel size 5, stride 2) designed to identify initial impact spikes.
  * Activation: ReLU
  * Max Pooling 1D (kernel size 2) to compress the timeline.
* **Layer 2 (The "Context" Detector):** * 1D Convolution (16 filters, kernel size 3, stride 1) to learn sequential patterns from the initial spikes.
  * Activation: ReLU
* **Pooling Strategy:** 1D Global Average Pooling (`AdaptiveAvgPool1d`) to crush the temporal dimension down to a single value per feature map, minimizing memory overhead.
* **Output Layer:** Fully Connected Linear layer mapping the 16 extracted features to 2 output neurons (Binary Classification: Fall vs. ADL).

### Local Training Hyperparameters
The following hyperparameters were locked in during the Federated Learning fine-tuning phase:

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Balances the memory limits of wearable devices with stable gradient updates during DP noise injection. |
| **Local Epochs** | 3 | Prevents catastrophic overfitting on the severely capped local client datasets. |
| **Learning Rate** | 0.001 | Ensures smooth convergence without violently disrupting the globally aggregated weights. |
| **Optimizer** | Adam / SGD | Adam for plaintext pretraining; SGD during DP training for gradient stability. |
| **Loss Function** | Cross-Entropy | Weighted `[1.0, 3.0]` to heavily penalize missed fall events. |

##  Dataset
This project uses the **SisFall Dataset**, a comprehensive dataset of falls and Activities of Daily Living (ADLs) acquired with a wearable sensor. Link to original paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/
* **Subjects:** 38 total (15 elderly, 23 young adults).
* **Pretraining Pool:** 14 subjects used for plaintext feature extraction.
* **Federated Pool:** 22 subjects used for secure FL simulation.

### Statistical Heterogeneity (Handling Non-IID Data)
In real-world Federated Learning deployments, biometric data is strictly Non-IID (Non-Independent and Identically Distributed). For example, a 20-year-old student walks and falls with entirely different kinetic force than an 80-year-old patient. 

* **The Challenge:** To quantify this data drift, we mapped the **Earth Mover's Distance (EMD)** across the FL clients. EMD measures the mathematical "cost" of turning one client's data distribution into another's. As shown in the project's EMD Heatmap, there is severe statistical heterogeneity between subjects, with distance scores frequently exceeding 200 between specific client pairs (indicated by the dark red zones). 

![Earth Mover's Distance Heatmap showing Non-IID data across 15 clients](images/emd_heatmap.png)

* **Our Solution's Proof:** Highly Non-IID data typically causes catastrophic weight divergence during server aggregation. The fact that our global model still seamlessly converges to a high F1 Score—even while battling Paillier quantization error and Opacus noise—proves that our Transfer Learning pretraining and robust 1D-CNN architecture successfully generalize across a highly diverse, heterogeneous population.

> **Note:** The dataset must be downloaded separately and placed in the `data/` directory.

## Hardware Feasibility 
To contextualize the system constraints recorded during the simulation, the computational footprint of this framework was designed to operate comfortably within the limits of commercially available wearable architectures. 

| Hardware Profile | Typical RAM | Network / Connectivity | Viability for this Framework |
| :--- | :--- | :--- | :--- |
| **High-End Smartwatch** <br>*(e.g., Apple Watch Series 11)* | 1 GB | Wi-Fi / LTE / BLE 5.3 | **Over-qualified.** Can easily run the model and transmit encrypted payloads with negligible battery drain. |
| **Low-Power Fitness Tracker** <br>*(e.g., Device using an ARM Cortex-M4)* | 256 KB - 512 KB | BLE (Bluetooth Low Energy) | **Highly Viable.** The ~500-parameter 1D-CNN easily fits within the strict RAM constraints of always-on fitness bands. |
| **Our Simulation Environment** | *400 MB (Python/PyTorch Overhead)* | *Localhost (Simulated)* | *Note: The ~400 MB RAM tracked in this simulation reflects the heavy Python/PyTorch backend overhead required for simulation. A compiled bare-metal edge-deployment (e.g., C++) would shrink the active memory footprint to mere kilobytes.* |

#### Network Tax & Payload Size
Wearable devices rely heavily on Bluetooth Low Energy (BLE), where transmitting large amounts of data severely degrades battery life by keeping the radio antenna active. 
* **The Homomorphic Encryption Challenge:** HE algorithms (like Paillier) suffer from ciphertext expansion, where a single parameter expands massively in size when encrypted. 
* **Our Solution:** By intentionally restricting the 1D-CNN to 562 parameters, the plaintext model size is roughly 0.003 MB. Even factoring in standard ciphertext expansion under HE, the resulting encrypted payload transmitted to the server remained around 0.4 MB. Given that the practical throughput of BLE 5.0 is approximately 175 KB/s, this payload falls well within the universally acceptable threshold for IoT devices (< 1 MB). It requires only 1 to 3 seconds of active radio transmission time per round, preventing connection drops and perfectly preserving the edge device's battery life.

## Repository Structure

├── data/                       # SisFall dataset (not included in repo)
├── results/                    # JSON files for the 4 experiments
├── client.py                   # Flower client setup
├── server.py                   # Custom strategy for HE aggregation and metric weighting
├── model.py                    # 1D-CNN architecture definitions and local training loop
├── 01_run_simulation.ipynb     # Flower simulation execution script
├── 02_data_analysis.ipynb      # Visualizations and analysis from simulations
└── README.md



