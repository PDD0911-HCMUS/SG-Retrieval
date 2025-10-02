import re
import matplotlib.pyplot as plt

# Lists to hold component losses per epoch
train_epochs, train_total, train_info_nce, train_cosine = [], [], [], []
val_epochs, val_total, val_info_nce, val_cosine = [], [], [], []

# Parse the log file for summary lines
log_path = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Controller/IRESGController/work_dir/LOGGER_2025-06-25_08-56-07.log'
with open(log_path, 'r') as f:
    for line in f:
        # Training summary
        m_train = re.search(
            r'Epoch (\d+) - Average Training Loss: ([0-9.]+) - info_nce: ([0-9.]+) - cosine_sim: ([0-9.]+)',
            line
        )
        if m_train:
            e, t, i, c = m_train.groups()
            train_epochs.append(int(e))
            train_total.append(float(t))
            train_info_nce.append(float(i))
            train_cosine.append(float(c))
        # Validation summary
        m_val = re.search(
            r'Epoch (\d+) - Validation Loss: ([0-9.]+) - info_nce: ([0-9.]+) - cosine_sim: ([0-9.]+)',
            line
        )
        if m_val:
            e, t, i, c = m_val.groups()
            val_epochs.append(int(e))
            val_total.append(float(t))
            val_info_nce.append(float(i))
            val_cosine.append(float(c))

# Plot component losses
plt.figure(figsize=(8, 5))
# Training components
plt.plot(train_epochs, train_total, marker='o', label='Train Total Loss')
plt.plot(train_epochs, train_info_nce, marker='o', label='Train InfoNCE Loss')
plt.plot(train_epochs, train_cosine, marker='o', label='Train CosineSim Loss')
# Validation components
plt.plot(val_epochs, val_total, marker='s', label='Val Total Loss')
plt.plot(val_epochs, val_info_nce, marker='s', label='Val InfoNCE Loss')
plt.plot(val_epochs, val_cosine, marker='s', label='Val CosineSim Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Component Losses over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
