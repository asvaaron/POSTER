import re
import matplotlib.pyplot as plt


def parse_log(log_file):

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(log_file, "r") as f:
        for line in f:
            if "Train time" in line:
                epoch = int(re.search(r"Epoch (\d+)", line).group(1))
                loss = float(re.search(r"Loss:\s*([0-9.]+)", line).group(1))
                acc = re.search(r"Training accuracy:([0-9.]+)", line).group(1)
                acc = float(acc.rstrip("."))

                epochs.append(epoch)
                train_loss.append(loss)
                train_acc.append(acc)


            elif "Validation accuracy" in line:
                loss = float(re.search(r"Loss:([0-9.]+)", line).group(1))
                acc = re.search(r"Validation accuracy:([0-9.]+)", line).group(1)
                acc = float(acc.rstrip(","))

                val_loss.append(loss)
                val_acc.append(acc)

    return epochs, train_loss, val_loss, train_acc, val_acc


file_name = "log/training_local_base_simple.log"

epochs1, train_loss1, val_loss1, train_acc1, val_acc1 = parse_log(file_name)

file_name2 = "log/training_local_base_enhanced_v1.log"

epochs2, train_loss2, val_loss2, train_acc2, val_acc2 = parse_log(file_name2)

# Define max of Epochs
MAX_EPOCH = 250
# Define model size name
model_size_name = "Base"

filtered_epochs = [e for e in epochs1 if e <= MAX_EPOCH]
idx = len(filtered_epochs)
# Training Loss
plt.figure()
plt.plot(epochs1[:idx], train_loss1[:idx], label="Train Loss No Changes")
plt.plot(epochs2[:idx], train_loss2[:idx], label="Train Loss Changes")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss per Epoch {model_size_name}")
plt.legend()
plt.show()

# Validation Loss
plt.figure()
plt.plot(epochs1[:idx], val_loss1[:idx], label="Validation Loss No Changes")
plt.plot(epochs2[:idx], val_loss2[:idx], label="Validation Loss Changes")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Validation Loss per Epoch {model_size_name}")
plt.legend()
plt.show()

# Training Accuracy
plt.figure()
plt.plot(epochs1[:idx], val_acc1[:idx], label="Training Accuracy No Changes")
plt.plot(epochs2[:idx], val_acc2[:idx], label="Training Accuracy Changes")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Training Accuracy per Epoch {model_size_name}")
plt.legend()
plt.show()


# Validation Accuracy
plt.figure()
plt.plot(epochs1[:idx], train_acc1[:idx], label="Validation Accuracy No Changes")
plt.plot(epochs2[:idx], train_acc2[:idx], label="Validation Accuracy Changes")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Validation Accuracy per Epoch {model_size_name}")
plt.legend()
plt.show()