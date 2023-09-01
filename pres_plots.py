# Plotting the augmentations

import matplotlib.pyplot as plt
# After 50 epochs 

# show what the defaults were for those augmentations 

# Example mAP values (replace these with your actual data)
methods = ['No Augmentation', 'Blur', 'Horizontal Flipping', 'Random Sharpness']
# method A BLUR: train = 0.9935017680370775, val =0.7645522951549639
# flipping : train = 0.9803945972730336, val = 0.8155913197540258
# sharpness: train = 0.9957616060035979, val = 0.7625355412613961
train_mAP = [1.0, 0.99, 0.98, 1.0]  # mAP values during training
val_mAP = [0.77, 0.76, 0.82, 0.76]    # mAP values during validation

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = range(len(methods))

# Bar plots for training and validation mAP
train_bars = ax.bar(index, train_mAP, bar_width, label='Training mAP')
val_bars = ax.bar([i + bar_width for i in index], val_mAP, bar_width, label='Validation mAP')

ax.set_xlabel('Augmentation Methods')
ax.set_ylabel('Mean Average Precision (mAP)')
ax.set_title('Mean Average Precision during Training and Validation')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(methods)
# ax.legend()

# Place the legend to the right of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Display the plot
plt.tight_layout()
# plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('mAP_augmentations.png')
