import os
import glob

def delete_files_with_keyword(root_folder, keyword):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if keyword in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == "__main__":
    root_folder = "/home/sicily/cv4e_cagedbird_ID/data/high"  # Replace this with the actual root folder path
    keyword = "_random_"
    delete_files_with_keyword(root_folder, keyword)
