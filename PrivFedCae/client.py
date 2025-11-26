"""
CLIENT APP — Simplified for Federated CAE training with Malware Detection
Simplified detection: any sample with reconstruction error above most benign errors is flagged as malware
"""

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
import pandas as pd
import numpy as np


# =============================================
# Dataset Definition
# =============================================
class PowerTraceDataset(Dataset):
    """Creates overlapping sliding windows from power trace sequence"""
    def __init__(self, power_values, window_size=100, overlap=0.5):
        self.window_size = window_size
        self.power_values = power_values
        self.stride = int(window_size * (1 - overlap))
        self.num_windows = (len(power_values) - window_size) // self.stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.power_values[start:end]
        return torch.FloatTensor(window).unsqueeze(0)  # (1, window_size)


# =============================================
# Model Definition
# =============================================
class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(16, 1, 5, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =============================================
# Data Loading Utilities
# =============================================
def normalize_power_trace(power_values):
    """Normalize power values to [0, 1] range"""
    power_min, power_max = power_values.min(), power_values.max()
    return (power_values - power_min) / (power_max - power_min + 1e-8)


def load_local_benign_data(device_id, csv_path="PrivFedCae/benign_power_traces.csv"):
    df = pd.read_csv(csv_path)
    device_df = df[df["device_id"] == device_id]
    if len(device_df) == 0:
        raise ValueError(f"No benign data found for device {device_id}")

    power_values = device_df["power_consumption_mW"].values
    normalized = normalize_power_trace(power_values)
    dataset = PowerTraceDataset(normalized, window_size=100, overlap=0.5)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader


def load_local_malware_data(device_id, csv_path="PrivFedCae/malicious_power_traces.csv"):
    try:
        df = pd.read_csv(csv_path)
        device_df = df[df["device_id"] == device_id]
        if len(device_df) == 0:
            print(f"⚠ No malware data available for device {device_id}")
            return None

        power_values = device_df["power_consumption_mW"].values
        normalized = normalize_power_trace(power_values)
        dataset = PowerTraceDataset(normalized, window_size=100, overlap=0.5)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        #print(f"✓ Loaded {len(device_df)} malware samples for device {device_id}")
        return loader

    except FileNotFoundError:
        print(f"⚠ Malware data file not found: {csv_path}")
        return None


# =============================================
# Training & Evaluation
# =============================================
def train(model, loader, epochs=3, lr=1e-3, noise_multiplier=1.0, max_grad_norm=1.0):
    """Train model on local benign data with differential privacy"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(loader.dataset)
        #print(f"[Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Training complete with ε = {epsilon:.2f}")
    return avg_loss


def evaluate(model, loader):
    """Evaluate model on benign test data"""
    criterion = nn.MSELoss()
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            loss_total += criterion(outputs, batch).item() * batch.size(0)
    return loss_total / len(loader.dataset)


# =============================================
# Simplified Malware Detection Logic
# =============================================
def compute_reconstruction_errors(model, loader):
    """Return reconstruction errors for each sample as plain python floats."""
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    errors = []

    if loader is None:
        return np.array([])

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            batch_errors = torch.mean(
                criterion(outputs, batch), dim=(1, 2)
            ).cpu().numpy()
            errors.extend(batch_errors)

    # Ensure float dtype
    return np.array(errors, dtype=float)


def simple_malware_detection(model, benign_loader, malware_loader, percentile=95):
    """
    Simplified detection:
    - Threshold = percentile of benign errors (e.g., p95)
    - Returns only JSON-serializable metrics (ints/floats) for Flower
    """
    # handle missing loaders gracefully
    benign_errors = compute_reconstruction_errors(model, benign_loader)
    malware_errors = compute_reconstruction_errors(model, malware_loader)

    # If no malware samples, return safe defaults
    if len(malware_errors) == 0:
        return {
            "detection_threshold": float(np.percentile(benign_errors, percentile)) if len(benign_errors) else 0.0,
            "malware_anomalies_detected": 0,
            "malware_anomaly_rate": 0.0,
            "malware_mean_error": float(np.mean(malware_errors)) if len(malware_errors) else 0.0,
            "malware_min_error": float(np.min(malware_errors)) if len(malware_errors) else 0.0,
            "malware_max_error": float(np.max(malware_errors)) if len(malware_errors) else 0.0,
            "benign_mean_error": float(np.mean(benign_errors)) if len(benign_errors) else 0.0,
        }

    # compute threshold and flags
    threshold = float(np.percentile(benign_errors, percentile)) if len(benign_errors) else float(np.mean(malware_errors))
    flags = malware_errors > threshold

    # convert to JSON-serializable scalars
    anomalies = int(np.sum(flags))
    anomaly_rate = float(100.0 * anomalies / len(malware_errors))
    malware_mean = float(np.mean(malware_errors))
    malware_min = float(np.min(malware_errors))
    malware_max = float(np.max(malware_errors))
    benign_mean = float(np.mean(benign_errors)) if len(benign_errors) else 0.0

    # Print debugging info (keeps your console logs)
    print("\n=== Simple Malware Detection ===")
    print(f"Benign Mean Error: {benign_mean:.6f}")
    print(f"Malware Mean Error: {malware_mean:.6f}")
    print(f"Detection Threshold (p{percentile}): {threshold:.6f}")
    print(f"Flagged Malware Samples: {anomalies} / {len(malware_errors)}")

    # show flagged indices for debugging only (do NOT return the array)
    flagged_indices = np.where(flags)[0]
    for idx in flagged_indices:
        print(f"  Sample {int(idx)} error: {float(malware_errors[int(idx)]) :.6f}")

    # return only serializable scalars
    return {
        "detection_threshold": threshold,
        "malware_anomalies_detected": anomalies,
        "malware_anomaly_rate": anomaly_rate,
        "malware_mean_error": malware_mean,
        "malware_min_error": malware_min,
        "malware_max_error": malware_max,
        "benign_mean_error": benign_mean,
    }



# =============================================
# Flower Client
# =============================================
class CAEClient(NumPyClient):
    """Federated client with simplified malware detection"""

    def __init__(self, model, train_loader, test_loader, malware_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.malware_loader = malware_loader
        self.threshold = None

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(
                self.model.state_dict().keys(),
                [torch.tensor(p) for p in parameters]
            )
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Local training round"""
        self.set_parameters(parameters)
        loss = train(
            self.model,
            self.train_loader,
            epochs=50,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )
        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": loss}

    def evaluate(self, parameters, config):
        """
        Robust evaluate: returns (loss, n_examples, metrics)
        Always returns a valid tuple (no None). Serializes metrics to plain types.
        """
        try:
            # set model params from server
            self.set_parameters(parameters)

            # compute validation loss on benign test set
            loss = evaluate(self.model, self.test_loader)
            print(f"[Eval] Validation loss: {loss:.6f}")

            # base metrics
            metrics = {"val_loss": float(loss)}

            # if malware data exists, run detection and attach safe metrics
            if self.malware_loader is not None:
                detection_stats = simple_malware_detection(
                    self.model, self.test_loader, self.malware_loader, percentile=95
                )
                # detection_stats already contains plain scalars
                metrics.update(detection_stats)

            # number of examples used for evaluation must be an int
            n_examples = len(self.test_loader.dataset) if self.test_loader is not None else 0
            return float(loss), int(n_examples), metrics

        except Exception as e:
            # If any error occurs, log it and still return a safe response so the server
            # sees a "failure" but doesn't crash the whole aggregation process.
            print(f"⚠ Exception in client evaluate(): {e}")
            # Return fallback metrics (server will see this client as failed/low-quality)
            fallback_metrics = {
                "val_loss": float("nan"),
                "malware_anomalies_detected": 0,
                "malware_anomaly_rate": 0.0,
                "detection_threshold": 0.0,
            }
            n_examples = len(self.test_loader.dataset) if getattr(self, "test_loader", None) is not None else 0
            return float("nan"), int(n_examples), fallback_metrics

# =============================================
# Client Factory & App
# =============================================
def client_fn(context: Context):
    """Factory function to create client instances"""
    partition_id = context.node_config["partition-id"]
    print(f"\n Starting client for device_id {partition_id}")

    train_loader = load_local_benign_data(device_id=partition_id)
    test_loader = load_local_benign_data(device_id=partition_id)
    malware_loader = load_local_malware_data(device_id=partition_id)

    model = Conv1dAutoencoder()
    client = CAEClient(model, train_loader, test_loader, malware_loader=malware_loader)
    return client.to_client()


app = ClientApp(client_fn=client_fn)
