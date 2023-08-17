import os
import shutil

def delete_random_image_folders(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            if folder_name == "random_images":
                try:
                    shutil.rmtree(folder_path)  # Remove the entire folder and its contents
                    print(f"Deleted: {folder_path}")
                except Exception as e:
                    print(f"Error deleting {folder_path}: {e}")

# Specify the root directory where you want to search for 'random_images' folders
root_directory = "/home/sicily/cv4e_cagedbird_ID/data/high"

delete_random_image_folders(root_directory)