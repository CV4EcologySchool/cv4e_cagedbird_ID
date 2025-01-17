import os

def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def bytes_to_gb(size_in_bytes):
    return size_in_bytes / (1024 ** 3)

# Paths to the folders
folder1 = '/home/home01/bssbf/cv4e_cagedbird_ID/data2'
folder2 = '/home/home01/bssbf/cv4e_cagedbird_ID/test_con2'

# Get the sizes of the folders
size1 = get_folder_size(folder1)
size2 = get_folder_size(folder2)

# Convert sizes to GB
size1_gb = bytes_to_gb(size1)
size2_gb = bytes_to_gb(size2)

# Calculate the total size in GB
total_size_gb = size1_gb + size2_gb

print(f"The total size of the folders is {total_size_gb:.2f} GB.")