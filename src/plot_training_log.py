import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_training_log(log_file):
    # Read the log file
    epochs = []
    val_ndcg = []
    val_hr = []
    test_ndcg = []
    test_hr = []

    with open(log_file, 'r') as file:
        for line in file:
            if line.startswith('epoch'):
                continue
            parts = line.split()
            epoch = int(parts[0])
            val_ndcg_value = float(parts[1].strip('(),np.float64'))
            val_hr_value = float(parts[2].strip('(),np.float64'))
            test_ndcg_value = float(parts[3].strip('(),np.float64'))
            test_hr_value = float(parts[4].strip('(),np.float64'))

            epochs.append(epoch)
            val_ndcg.append(val_ndcg_value)
            val_hr.append(val_hr_value)
            test_ndcg.append(test_ndcg_value)
            test_hr.append(test_hr_value)

    # Plot the metrics
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_ndcg, label='Validation NDCG')
    plt.plot(epochs, test_ndcg, label='Test NDCG')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title('NDCG over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_hr, label='Validation HR')
    plt.plot(epochs, test_hr, label='Test HR')
    plt.xlabel('Epoch')
    plt.ylabel('HR')
    plt.title('HR over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_training_log.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    plot_training_log(log_file)