# PrivFedCAE

### Privacy-Preserving Federated Autoencoder for IoT Malware Detection

> ⚠️ **Work in Progress:** This project is currently under development. Features may change.

**PrivFedCAE** is a lightweight anomaly detection system for IoT devices. It detects malware (like Rowhammer, Spectre, or Covert Channels) by analyzing device power consumption patterns—without ever sending raw data to a central server.

## 🚀 How It Works

1. **Local Training:** IoT devices individually train a lightweight **Convolutional Autoencoder** on their own benign power usage data.
2. **Federated Learning:** Using the **Flower** framework, devices send only model updates (gradients) to a central server—never their raw data.
3. **Privacy:** We apply **Differential Privacy** (via Opacus) to these updates, adding noise to mathematically guarantee user privacy.
4. **Detection:** The global model is sent back to devices. If a device's power pattern shows high reconstruction error, it is flagged as a potential malware anomaly.

## 📂 Project Structure

- **`SDG.py`**: Synthetic Data Generator. Creates realistic IoT power traces (idle, normal, scanning, loading) and injects malware anomalies.
- **`client.py`**: The Flower client script. Handles local training, differential privacy, and malware detection logic.
- **`server.py`**: The central aggregator. Averages model updates from clients to build the global model.
- **`CAE.py`**: PyTorch definition of the Conv1d Autoencoder.

## 🛠️ Quick Start

### 1. Setup Environment
```bash
# Clone the repo
git clone https://github.com/yourusername/PrivFedCAE.git
cd PrivFedCAE

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data
Create the benign and malicious power trace datasets:
```bash
python SDG.py
# Outputs: benign_power_traces.csv, malicious_power_traces.csv
```

### 3. Run the Simulation
You can simulate the federated learning process (server + 5 clients) locally:

```bash
# run the setup using the command
flwr run .
```

## ✅ Current Status

- [x] Synthetic Power-Trace Generator (Benign + Malicious)
- [x] Convolutional Autoencoder (PyTorch)
- [x] Federated Learning Setup (Flower)
- [x] Differential Privacy Integration (Opacus)


## 📄 License

MIT License
