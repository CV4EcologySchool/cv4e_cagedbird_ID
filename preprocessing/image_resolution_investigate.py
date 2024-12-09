import os
from PIL import Image

def get_image_resolution(image_path):
    """Get the resolution (width * height) of an image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width * height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0

def sort_folders_by_image_resolution(base_folder):
    """Sort subfolders by the average image resolution of the images they contain."""
    folder_resolutions = {}
    
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            total_resolution = 0
            image_count = 0
            
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                    resolution = get_image_resolution(file_path)
                    if resolution > 0:
                        total_resolution += resolution
                        image_count += 1
            
            if image_count > 0:
                avg_resolution = total_resolution / image_count
                folder_resolutions[folder] = avg_resolution
    
    # Sort folders by average resolution
    sorted_folders = sorted(folder_resolutions.items(), key=lambda x: x[1], reverse=True)
    
    print("Folders sorted by average image resolution:")
    for folder, resolution in sorted_folders:
        print(f"{folder}: {resolution:.2f} pixels")

# Example usage
base_folder = "path/to/your/folder"  # Replace with the path to your folder
sort_folders_by_image_resolution(base_folder)
