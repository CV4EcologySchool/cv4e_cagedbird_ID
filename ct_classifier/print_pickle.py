import pickle

# Path to your pickle file
pickle_file_path = '/home/home01/bssbf/cv4e_cagedbird_ID/data/class_mapping_29.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as f:
    class_mapping = pickle.load(f)

# View the first few items in the class mapping to verify
# print(list(class_mapping.items())[:10])  # Print first 10 key-value pairs
print(class_mapping)