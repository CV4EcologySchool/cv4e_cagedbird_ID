from markdown_pdf import MarkdownPdf, Section

# Path to your local Markdown file
markdown_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/species_list.md'

# Read the content of the Markdown file
with open(markdown_file_path, 'r', encoding='utf-8') as file:
    markdown_content = file.read()

# Create the PDF
pdf = MarkdownPdf()
pdf.meta["title"] = 'Title'
pdf.add_section(Section(markdown_content, toc=False))
pdf.save('species_list_april_25.pdf')