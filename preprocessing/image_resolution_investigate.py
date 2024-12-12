import os
from PIL import Image
import csv

def get_image_resolution(image_path):
    """Get the resolution (width * height) of an image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width * height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0

def sort_folders_by_average_pixel_count(base_folder, output_csv):
    """Sort subfolders by the average pixel count of the images they contain and save results to a CSV."""
    folder_data = []
    
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            total_pixels = 0
            image_count = 0
            
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                    resolution = get_image_resolution(file_path)
                    if resolution > 0:
                        total_pixels += resolution
                        image_count += 1
            
            avg_pixels = total_pixels / image_count if image_count > 0 else 0
            folder_data.append((folder, image_count, avg_pixels))
    
    # Sort folders by average pixel count
    sorted_folders = sorted(folder_data, key=lambda x: x[2], reverse=True)
    
    # Add rankings
    ranked_folders = [(rank + 1, folder, image_count, avg_pixels) 
                      for rank, (folder, image_count, avg_pixels) in enumerate(sorted_folders)]
    
    print("Folders ranked by average image pixel count:")
    for rank, folder, image_count, avg_pixels in ranked_folders:
        print(f"Rank {rank}: {folder} - {image_count} images, {avg_pixels:.2f} average pixels")
    
    # Write results to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Folder Name", "Image Count", "Average Pixel Count (pixels)"])
        writer.writerows(ranked_folders)
    
    print(f"Results saved to {output_csv}")


# Example usage
base_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"  # Replace with the path to your folder
output_csv = "preprocessing/species_by_average_pixel_count.csv"  # Specify the output CSV file name
sort_folders_by_average_pixel_count(base_folder, output_csv)

