import os
import shutil

def rename_files_in_subfolders(root_dir):
    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            random_images_dir = os.path.join(folder_path, "random_images")
            if os.path.isdir(random_images_dir):
                for i,filename in enumerate(os.listdir(random_images_dir)):
                    file_path = os.path.join(random_images_dir, filename)
                    if os.path.isfile(file_path):
                        new_filename = f"{foldername}_random_{i}.jpg"
                        new_file_path = os.path.join(folder_path, new_filename)
                        shutil.copy(file_path, new_file_path)
                        print(f"Renamed: {filename} -> {new_filename}")

# Replace 'root_directory_path' with the path to the root directory containing the subfolders
root_directory_path = '/home/sicily/cv4e_cagedbird_ID/data/high'
rename_files_in_subfolders(root_directory_path)
