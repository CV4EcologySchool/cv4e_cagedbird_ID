import matplotlib.pyplot as plt
import numpy as np

# Example mAP values (replace these with your actual data)
methods = ['No Aug Unbalanced', 'Blur Unbalanced', 'Horizontal Flipping Unbalanced', 'Random Sharpness Unbalanced', 'All Aug Unbalanced', 'All Aug Upsampled']
train_mAP = [1.0, 0.99, 0.98, 1.0, 0.97, 0.97]  # mAP values during training
val_mAP = [0.77, 0.76, 0.82, 0.76, 0.77, 0.79]    # mAP values during validation

# Apply logarithmic transformation to the data
train_mAP_log = np.log10(train_mAP)
val_mAP_log = np.log10(val_mAP)

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(methods))

# Calculate the x-coordinates for both training and validation bars
combined_x = index + bar_width / 2

# Bar plots for training and validation mAP
train_bars = ax.bar(combined_x, train_mAP_log, bar_width, label='Training mAP (log scale)')
val_bars = ax.bar(combined_x, val_mAP_log, bar_width, label='Validation mAP (log scale)', bottom=train_mAP_log)  # Stack validation bars on top of training bars

ax.set_xlabel('Augmentation Methods')
ax.set_ylabel('Log10 of Mean Average Precision (mAP)')
ax.set_title('Mean Average Precision during Training and Validation (log scale)')
ax.set_xticks([])  # Remove x-axis ticks
ax.set_xticklabels([])  # Remove x-axis labels

# Place the legend to the right of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add labels in the middle of each bar
for i, label in enumerate(methods):
    ax.text(combined_x[i], (train_mAP_log[i] + val_mAP_log[i]) / 2, label, ha='center', va='bottom', rotation=45)

# Display the plot
plt.tight_layout()
plt.savefig('mAP_augmentations_log_scale.png')
plt.show()
