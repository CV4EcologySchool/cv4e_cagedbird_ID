import os
import pickle

def create_class_mapping_from_folder(folder_path):
    """
    Create a class mapping pickle file from the subfolders of a given folder.
    Each subfolder is considered a category (class).
    
    Args:
    - folder_path (str): Path to the folder containing subfolders for each class.
    
    Returns:
    - class_mapping (dict): A dictionary mapping class names to indices.
    """
    # List the subfolders in the given directory
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    
    # Create a mapping from class names (subfolder names) to indices
    class_mapping = {subfolder: idx for idx, subfolder in enumerate(subfolders)}
    
    # Save the mapping to a pickle file
    pickle_file = os.path.join(folder_path, 'class_mapping_29.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"Class mapping saved to {pickle_file}")
    return class_mapping

# Example usage
folder_path = '/home/home01/bssbf/cv4e_cagedbird_ID/data'  # Replace with your folder path
class_mapping = create_class_mapping_from_folder(folder_path)
