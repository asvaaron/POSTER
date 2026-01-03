import re
import matplotlib.pyplot as plt

epochs = []
train_loss = []
val_loss = []

file_name = "log/training_base_no_weights.log"

with open("log/training_local_base_enhanced_v1.log", "r") as f:
    for line in f:
        if "Train time" in line:
            epoch = int(re.search(r"Epoch (\d+)", line).group(1))
            loss = float(re.search(r"Loss:\s*([0-9.]+)", line).group(1))
            epochs.append(epoch)
            train_loss.append(loss)

        elif "Validation accuracy" in line:
            loss = float(re.search(r"Loss:([0-9.]+)", line).group(1))
            val_loss.append(loss)

# Define max of Epochs
MAX_EPOCH = 200

filtered_epochs = [e for e in epochs if e <= MAX_EPOCH]
idx = len(filtered_epochs)

# Plot file
plt.figure()
plt.plot(epochs[:idx], train_loss[:idx], label="Train Loss")
plt.plot(epochs[:idx], val_loss[:idx], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()