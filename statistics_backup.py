"""
Count the number of annotations in the dataset
"""
import os
from collections import Counter

# Set the directory where the label folders are located
base_dir = './fuse_dataset/labels/rgb'  # Replace with the path to your dataset
label_folders = ['train', 'val', 'test']

# Initialize a counter for all classes
class_counter = Counter()

# Define class indexes corresponding to your dataset labels
# Adjust the index numbers according to your dataset class indexes
class_indexes = {'yacht': 0, 'speedboat': 1, 'sailboat': 2, 'fishboat': 3}

for folder in label_folders:
    folder_path = os.path.join(base_dir, folder)
    # Go through all the txt files in each label folder
    for label_file in os.listdir(folder_path):
        if label_file.endswith('.txt') and not label_file.startswith('classes'):
            with open(os.path.join(folder_path, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    # Increment the count for this class
                    class_name = [name for name, index in class_indexes.items() if index == class_id][0]
                    class_counter[class_name] += 1

# Print the count for each class
for class_name, count in class_counter.items():
    print(f'Class {class_name}: {count}')