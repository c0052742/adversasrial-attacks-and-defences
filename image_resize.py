import os
import cv2
import argparse

def resize_images_in_folder(input_folder_path, target_size):
    for filename in os.listdir(input_folder_path):
        if not filename.lower().endswith('.jpg') and not filename.lower().endswith('.jpeg') and not filename.lower().endswith('.png'):
            continue
        input_path = os.path.join(input_folder_path, filename)
        img = cv2.imread(input_path)
        img_resized = cv2.resize(img, target_size)
        cv2.imwrite(input_path, img_resized)

parser = argparse.ArgumentParser(description='Resize images in a folder.')
parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing images')
parser.add_argument('--target_size', type=str, required=True, help='Target size for resizing in format width,height like 1245,375')
args = parser.parse_args()

# Extract target size
target_size = tuple(map(int, args.target_size.split(',')))

# Resize images
resize_images_in_folder(args.input_folder, target_size)
