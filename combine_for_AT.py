import os
import random
import shutil
import argparse

def combine_and_split_data(input_folders, combined_folder, train_folder, val_folder, train_ratio):
    # Create combined folder if it doesn't exist
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)
    if not os.path.exists(os.path.join(combined_folder, 'images')):
        os.makedirs(os.path.join(combined_folder, 'images'))
    if not os.path.exists(os.path.join(combined_folder, 'labels')):
        os.makedirs(os.path.join(combined_folder, 'labels'))

    # Combine images and labels into the combined folder
    for folder in input_folders:
        for file_name in os.listdir(os.path.join(folder, 'images')):
            if file_name.endswith('.png'):
                image_path = os.path.join(folder, 'images', file_name)
                label_path = os.path.join(folder, 'labels', file_name[:-4] + '.txt')
                combined_image_path = os.path.join(combined_folder, 'images', file_name)
                combined_label_path = os.path.join(combined_folder, 'labels', file_name[:-4] + '.txt')
                shutil.copy(image_path, combined_image_path)
                shutil.copy(label_path, combined_label_path)

    # Create train and validation folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(os.path.join(train_folder, 'images'))
        os.makedirs(os.path.join(train_folder, 'labels'))
    if not os.path.exists(val_folder):
        os.makedirs(os.path.join(val_folder, 'images'))
        os.makedirs(os.path.join(val_folder, 'labels'))

    # Randomly split combined data into train and validation sets
    all_files = os.listdir(os.path.join(combined_folder, 'images'))
    num_train_files = int(len(all_files) * train_ratio)
    train_files = set(random.sample(all_files, num_train_files))

    # Move files into train and validation folders
    for file_name in all_files:
        source_image_path = os.path.join(combined_folder, 'images', file_name)
        source_label_path = os.path.join(combined_folder, 'labels', file_name[:-4] + '.txt')
        if file_name in train_files:
            dest_image_path = os.path.join(train_folder, 'images', file_name)
            dest_label_path = os.path.join(train_folder, 'labels', file_name[:-4] + '.txt')
        else:
            dest_image_path = os.path.join(val_folder, 'images', file_name)
            dest_label_path = os.path.join(val_folder, 'labels', file_name[:-4] + '.txt')
        shutil.copy(source_image_path, dest_image_path)
        shutil.copy(source_label_path, dest_label_path)

parser = argparse.ArgumentParser(description='Combine and split data for adversarial training defence.')
parser.add_argument('--input_folders', type=str, nargs='+', required=True, help='Input folders containing images and labels')
parser.add_argument('--combined_folder', type=str, required=True, help='Output folder to store combined images and labels')
parser.add_argument('--train_folder', type=str, required=True, help='Output folder to store training data')
parser.add_argument('--val_folder', type=str, required=True, help='Output folder to store validation data')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training (default: 0.8)')
args = parser.parse_args()

# Call the function to combine and split the data
combine_and_split_data(args.input_folders, args.combined_folder, args.train_folder, args.val_folder, args.train_ratio)