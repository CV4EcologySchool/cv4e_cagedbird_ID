import os
import random
from PIL import Image

# Input paths and folder subset
main_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"  # Replace with your main folder path
output_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/species_examples"  # Replace with your desired output folder path, which I'll make
output_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/species_examples"  # Replace with your desired output folder path, which I'll make

# List of subfolders to include (subset of 46)
subset_folders  = [
    'af_bluebird', 'bali_myna', 'bc_hanging_parrot', 'bh_bulbul', 'bm_leafbird',
    'bt_laughingthrush', 'bw_leafbird', 'cc_laughing', 'cc_thrush', 'cg_magpie',
    'common_myna', 'crested_lark', 'crested_myna', 'ft_barbet', 'gf_leafbird',
    'gg_leafbird', 'hill_myna', 'hooded_butcherbird', 'hoopoe', 'hwamei',
    'jap_grosbeak', 'javan_sparrow', 'jb_pitta', 'jp_starling', 'lg_leafbird',
    'oh_thrush', 'om_robin', 'rb_leiothrix', 'rubythroat', 'rw_bulbul', 'sb_munia',
    'scarlet_minivet', 'se_mesia', 'sh_bulbul', 'spotted_dove', 'sum_laughingthrush',
    'swinhoes_white_eye', 'wc_laughingthrush', 'wh_munia', 'wr_munia', 'wr_shama',
    'yb_tit', 'zebra_dove', 'zebra_finch', 'Eurasian_jay', 'Eurasian_siskin'
]

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# A4 portrait width (in pixels, adjust DPI as needed)
a4_width = 2480  # 210mm at 300 DPI
a4_height = 3508  # Just for reference, we won't use the height in this case

# Process each subfolder in the subset
for subfolder in subset_folders:
    species_folder = os.path.join(main_folder, subfolder)
    species_name = subfolder  # Use folder name as the species name (or customize as needed)

    if not os.path.isdir(species_folder):
        print(f"Warning: Subfolder {subfolder} not found, skipping.")
        continue

    # Get all image files in the subfolder
    image_files = [f for f in os.listdir(species_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 3:
        print(f"Warning: Not enough images in {subfolder}, skipping.")
        continue

    # Randomly select 3 images
    selected_images = random.sample(image_files, 3)
    images = [Image.open(os.path.join(species_folder, img)) for img in selected_images]

    # Resize images to fit side by side in A4 width
    individual_width = a4_width // 3
    resized_images = [img.resize((individual_width, int(img.height * (individual_width / img.width)))) for img in images]

    # Create a new blank image for the panel
    max_height = max(img.height for img in resized_images)
    panel = Image.new('RGB', (a4_width, max_height), (255, 255, 255))

    # Paste resized images side by side
    x_offset = 0
    for img in resized_images:
        panel.paste(img, (x_offset, 0))
        x_offset += individual_width

    # Save the panel image
    output_path = os.path.join(output_folder, f"{species_name}.jpg")
    panel.save(output_path)

print("Process complete. Check the output folder for the images.")


# Redo for all species, not just the subsetted ones, This code will process every folder in the main folder, not just the subset

# Input paths and folder subset
main_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/data2"  # Replace with your main folder path
output_folder = "/home/home01/bssbf/cv4e_cagedbird_ID/species_examples2"  # Replace with your desired output folder path

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# A4 portrait width (in pixels, adjust DPI as needed)
a4_width = 2480  # 210mm at 300 DPI
a4_height = 3508  # Just for reference, we won't use the height in this case

# Process each subfolder in the main folder
for subfolder in os.listdir(main_folder):
    species_folder = os.path.join(main_folder, subfolder)
    species_name = subfolder  # Use folder name as the species name (or customize as needed)

    if not os.path.isdir(species_folder):
        print(f"Warning: Subfolder {subfolder} not found, skipping.")
        continue

    # Get all image files in the subfolder
    image_files = [f for f in os.listdir(species_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 3:
        print(f"Warning: Not enough images in {subfolder}, skipping.")
        continue

    # Randomly select 3 images
    selected_images = random.sample(image_files, 3)
    images = [Image.open(os.path.join(species_folder, img)) for img in selected_images]

    # Resize images to fit side by side in A4 width
    individual_width = a4_width // 3
    resized_images = [img.resize((individual_width, int(img.height * (individual_width / img.width)))) for img in images]

    # Create a new blank image for the panel
    max_height = max(img.height for img in resized_images)
    panel = Image.new('RGB', (a4_width, max_height), (255, 255, 255))

    # Paste resized images side by side
    x_offset = 0
    for img in resized_images:
        panel.paste(img, (x_offset, 0))
        x_offset += individual_width

    # Save the panel image
    output_path = os.path.join(output_folder, f"{species_name}.jpg")
    panel.save(output_path)

print("Process complete. Check the output folder for the images.")


