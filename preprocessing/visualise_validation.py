import json
import matplotlib.pyplot as plt

# Load COCO annotations from the JSON file
json_file_path = '/home/sicily/cv4e_cagedbird_ID/data/high/validation.json'
with open(json_file_path, 'r') as json_file:
    coco_annotations = json.load(json_file)

# Iterate through categories
# for category in coco_annotations["categories"]:
#     category_name = category["name"]
#     class_summary[category_name] +=1 

class_summary = {}

# Count instances for each category
for annotation in coco_annotations["annotations"]:
    category_id = f'{annotation["category_id"]}'# f' converts to a string
    # category = next(cat for cat in coco_annotations["categories"] if cat["id"] == category_id)
    # category_name = category["name"]
    if category_id in class_summary.keys(): 
        class_summary[category_id] += 1
    else: 
        class_summary [category_id] = 1

counts = class_summary.values()
categories = class_summary.keys()

print(categories)

# Convert the keys to integers and sort them
sorted_categories = sorted(categories, key=lambda x: int(x))

print (sorted_categories)

plt.bar(sorted_categories, counts)
plt.xlabel("Categories")
plt.ylabel("Counts")
plt.title("Distribution of images across species classes for validation data")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig('validation_data_categories.png')