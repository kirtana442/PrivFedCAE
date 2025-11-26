import matplotlib.pyplot as plt
import numpy as np
from client import Conv1dAutoencoder, compute_reconstruction_errors, PowerTraceDataset, load_local_benign_data, load_local_malware_data
from torch.utils.data import DataLoader
import torch

# Load a trained model (or train a fresh one)
model = Conv1dAutoencoder()
# ... train model on benign data ...

# Load benign test data
benign_loader = load_local_benign_data(device_id=0)  # Your loading function
benign_errors = compute_reconstruction_errors(model, benign_loader)

# Load malware test data
malware_loader = load_local_malware_data(device_id=0)  # Create this function
malware_errors = compute_reconstruction_errors(model, malware_loader)

# Plot histograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(benign_errors, bins=50, alpha=0.7, label='Benign', color='green')
plt.hist(malware_errors, bins=50, alpha=0.7, label='Malware', color='red')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error Distribution Overlap')

plt.subplot(1, 2, 2)
plt.boxplot([benign_errors, malware_errors], labels=['Benign', 'Malware'])
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Error Distribution Statistics')

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=150)
plt.show()

# Print statistics
print("=== BENIGN ERROR STATS ===")
print(f"Mean: {np.mean(benign_errors):.6f}")
print(f"Std:  {np.std(benign_errors):.6f}")
print(f"Min:  {np.min(benign_errors):.6f}")
print(f"Max:  {np.max(benign_errors):.6f}")
print(f"50th percentile: {np.percentile(benign_errors, 50):.6f}")
print(f"95th percentile: {np.percentile(benign_errors, 95):.6f}")
print(f"99th percentile: {np.percentile(benign_errors, 99):.6f}")

print("\n=== MALWARE ERROR STATS ===")
print(f"Mean: {np.mean(malware_errors):.6f}")
print(f"Std:  {np.std(malware_errors):.6f}")
print(f"Min:  {np.min(malware_errors):.6f}")
print(f"Max:  {np.max(malware_errors):.6f}")
print(f"50th percentile: {np.percentile(malware_errors, 50):.6f}")
print(f"95th percentile: {np.percentile(malware_errors, 95):.6f}")
print(f"99th percentile: {np.percentile(malware_errors, 99):.6f}")

# Check overlap
overlap_95 = np.sum(malware_errors < np.percentile(benign_errors, 95)) / len(malware_errors) * 100
print(f"\n=== OVERLAP ===")
print(f"Malware samples below 95th percentile of benign: {overlap_95:.2f}%")
