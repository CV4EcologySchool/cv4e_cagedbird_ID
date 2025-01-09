import json

# Load the annotations from test.json
with open('/home/home01/bssbf/cv4e_cagedbird_ID/test/test.json', 'r') as f:
    annotations = json.load(f)

# Check the type of annotations
print(f"Type of annotations: {type(annotations)}")

# If it's a dictionary, print the first 5 key-value pairs
if isinstance(annotations, dict):
    # Print the first 5 key-value pairs of the dictionary
    for idx, (key, value) in enumerate(annotations.items()):
        if idx == 5:  # Stop after 5 entries
            break
        print(f"Key: {key}, Value: {value}")
else:
    # If it's a list, slice the first 5 elements
    for annotation in annotations[:5]:
        print(annotation)
