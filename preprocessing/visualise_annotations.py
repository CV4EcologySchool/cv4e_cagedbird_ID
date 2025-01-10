import os
import matplotlib.pyplot as plt

def count_images_in_folder(folder_path):
    image_count = 0
    for root, _, files in os.walk(folder_path):
        image_count += len([f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return image_count

root_directory = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"
subfolders = [f.path for f in os.scandir(root_directory) if f.is_dir()]

image_counts = []

for folder in subfolders:
    images_in_folder = count_images_in_folder(folder)
    image_counts.append(images_in_folder)

# Sort subfolders based on image counts
sorted_subfolders = [folder for _, folder in sorted(zip(image_counts, subfolders), reverse=True)]

plt.figure(figsize=(10, 6))
colors = plt.cm.tab20.colors  # Choose a colormap with enough distinct colors

bars = plt.bar(range(len(sorted_subfolders)), sorted(image_counts, reverse=True), color=colors)

# Set x-axis labels to empty strings (ticks only)
plt.xticks(range(len(sorted_subfolders)), [''] * len(sorted_subfolders))

plt.xlabel('Species')
plt.ylabel('Number of Images')
plt.title('Image counts per species')

# Removing legend
plt.legend().set_visible(False)

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("image_counts_plot.png")

print("Plot saved as 'image_counts_plot.png'.")
