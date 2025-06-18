import re
import matplotlib.pyplot as plt

# Initialize lists
non_10, non_20, non_50 = [], [], []
ed_10, ed_20, ed_50 = [], [], []

# Parse the log file
with open('/workspace/Controller/IRESGController/work_dir/LOGGER_2025-06-17_12-05-52.log', 'r') as f:
    for line in f:
        # Non-Editted recall
        m_non = re.search(r'non-Editted \| R@10: ([0-9.]+) \| R@20: ([0-9.]+) \| R@50: ([0-9.]+)', line)
        if m_non:
            non_10.append(float(m_non.group(1)))
            non_20.append(float(m_non.group(2)))
            non_50.append(float(m_non.group(3)))
        # Editted recall
        m_ed = re.search(r'Editted\s+\| R@10: ([0-9.]+) \| R@20: ([0-9.]+) \| R@50: ([0-9.]+)', line)
        if m_ed:
            ed_10.append(float(m_ed.group(1)))
            ed_20.append(float(m_ed.group(2)))
            ed_50.append(float(m_ed.group(3)))

# Align lengths
n = min(len(non_10), len(ed_10))
epochs = list(range(n))
non_10, non_20, non_50 = non_10[:n], non_20[:n], non_50[:n]
ed_10, ed_20, ed_50 = ed_10[:n], ed_20[:n], ed_50[:n]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, non_10, marker='o', label='Non-Editted R@10')
plt.plot(epochs, non_20, marker='o', label='Non-Editted R@20')
plt.plot(epochs, non_50, marker='o', label='Non-Editted R@50')
plt.plot(epochs, ed_10, marker='s', label='Editted R@10')
plt.plot(epochs, ed_20, marker='s', label='Editted R@20')
plt.plot(epochs, ed_50, marker='s', label='Editted R@50')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()