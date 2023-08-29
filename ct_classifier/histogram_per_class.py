# Create a dictionary to store scores for each class
class_scores = {class_label: [] for class_label in range(cfg['num_classes'])}

# Iterate through the validation data
for inputs, labels in dl_val:
    predictions = model(inputs)
    max_pred = predictions.max(dim=1).values
    
    for i in range(len(labels)):
        class_label = labels[i].item()
        class_scores[class_label].append(max_pred[i].item())

# Plot histograms for each class
num_bins = 50  # You can adjust this value based on your preference
plt.figure(figsize=(10, 6))

for class_label, scores in class_scores.items():
    plt.hist(scores, bins=num_bins, alpha=0.5, label=f'Class {class_label} Scores')

plt.xlabel('Class Scores')
plt.ylabel('Frequency')
plt.title('Histogram of Class Scores')
plt.legend()
# plt.show()
plt.savefig('Histogram_Scores_by_Class')
