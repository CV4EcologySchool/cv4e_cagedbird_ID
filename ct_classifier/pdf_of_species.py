import os
import pickle
import pandas as pd
import random
from PIL import Image
from PyPDF2 import PdfWriter, PdfReader
from io import BytesIO

# Path to the pickle file
pickle_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'

# Load the classes from the pickle file
with open(pickle_file_path, 'rb') as f:
    class_mapping = pickle.load(f)

# Path to the Excel file
excel_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/all_species_info.xlsx'

# Load the Excel file
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Folder containing species subfolders
species_folder = '/home/home01/bssbf/cv4e_cagedbird_ID/data2'

# Create a new PDF document using PyPDF2
pdf_writer = PdfWriter()

# Iterate over the class names and match with the Species Code in the Excel file
for class_name in class_mapping:
    species_info = df[df['Species Code'] == class_name]
    if not species_info.empty:
        scientific_name = species_info['Scientific Name'].values[0]
        common_name = species_info['Common Name'].values[0]
        indonesian_name = species_info['Indonesian Name'].values[0]
        iucn_status = species_info['IUCN Status'].values[0]
        population_status = species_info['Population Status'].values[0]
        
        # Create the text for the species info
        text = f"Class Name: {class_name}\n"
        text += f"Scientific Name: {scientific_name}\n"
        text += f"Common Name: {common_name}\n"
        text += f"Indonesian Name: {indonesian_name}\n"
        text += f"IUCN Status: {iucn_status}\n"
        text += f"Population Status: {population_status}\n\n"

        # Select a random image from the species folder
        species_path = os.path.join(species_folder, class_name)
        if os.path.isdir(species_path):
            species_images = [f for f in os.listdir(species_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if species_images:
                random_image = random.choice(species_images)
                image_path = os.path.join(species_path, random_image)

                # Open the image using PIL
                image = Image.open(image_path)

                # Save the image to a BytesIO object
                image_bytes = BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes.seek(0)

                # Insert the image into the PDF
                pdf_page = pdf_writer.add_blank_page(width=600, height=800)  # Adjust size accordingly
                pdf_page.merge_text(text)
                pdf_page.merge_image(image_bytes)  # Add image

# Save the PDF document
with open('species_information_with_images.pdf', 'wb') as output_pdf:
    pdf_writer.write(output_pdf)

print("PDF document 'species_information_with_images.pdf' has been created.")
