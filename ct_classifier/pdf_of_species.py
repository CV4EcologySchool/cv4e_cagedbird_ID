# Install the required packages
# pip install reportlab
# pip install pypdf

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
import pickle
import pandas as pd
import os
import random

# Path to the pickle file
pickle_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/ct_classifier/class_mapping_56.pickle'

# Load the classes from the pickle file
with open(pickle_file_path, 'rb') as f:
    class_mapping = pickle.load(f)

# Path to the Excel file
excel_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/all_species_info.xlsx'

# Load the Excel file
df = pd.read_excel(excel_file_path, engine='openpyxl')

# Create a PDF with reportlab
pdf_file_path = "species_information_reportlab.pdf"
c = canvas.Canvas(pdf_file_path, pagesize=letter)

# Set up the PDF layout
c.setFont("Helvetica", 10)

# Starting coordinates for the text
x, y = 72, 750

# Add the title
c.drawString(x, y, "Species Information\n")
y -= 20

# Path to the species folder
species_folder = '/home/home01/bssbf/cv4e_cagedbird_ID/data2'

# Iterate over the class names and match with the Species Code in the Excel file
for class_name in class_mapping:
    species_info = df[df['Species Code'] == class_name]
    
    if not species_info.empty:
        scientific_name = species_info['Scientific Name'].values[0]
        common_name = species_info['Common Name'].values[0]
        indonesian_name = species_info['Indonesian Name'].values[0]
        iucn_status = species_info['IUCN Status'].values[0]
        population_status = species_info['Population Status'].values[0]

        # Add species information (above the image)
        c.drawString(x, y, f"Class Name: {class_name}")
        y -= 15
        c.drawString(x, y, f"Scientific Name: {scientific_name}")
        y -= 15
        c.drawString(x, y, f"Common Name: {common_name}")
        y -= 15
        c.drawString(x, y, f"Indonesian Name: {indonesian_name}")
        y -= 15
        c.drawString(x, y, f"IUCN Status: {iucn_status}")
        y -= 15
        c.drawString(x, y, f"Population Status: {population_status}")
        y -= 25

        # Randomly select an image for the species (from the corresponding subfolder)
        species_subfolder = os.path.join(species_folder, class_name)
        if os.path.exists(species_subfolder):
            species_images = os.listdir(species_subfolder)
            if species_images:
                image_path = os.path.join(species_subfolder, random.choice(species_images))
                try:
                    # Draw the image (resize as needed)
                    c.drawImage(image_path, x + 100, y - 100, width=150, height=100)  # Adjust size and position as needed
                    y -= 120  # Space for the image
                except Exception as e:
                    print(f"Failed to add image for {class_name}: {e}")
        
        # Add space between species
        if y < 100:
            c.showPage()  # Add a new page if the text goes too low on the current page
            y = 750

# Save the PDF
c.save()

# Now use pypdf to manipulate the PDF
reader = PdfReader(pdf_file_path)
writer = PdfWriter()

# Add pages to the writer object (in case there are multiple pages)
for page in reader.pages:
    writer.add_page(page)

# Save the final PDF
final_pdf_path = "final_species_information.pdf"
with open(final_pdf_path, "wb") as f:
    writer.write(f)

print(f"PDF created and saved successfully as {final_pdf_path}")
