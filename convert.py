import os
import argparse
from PIL import Image

def convert_folder_kitti_to_yolo_v5(labels_folder_path, image_folder_path):
    classes = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, "Person_sitting": 4, "Cyclist": 5, "Tram": 6, "Misc": 7}
    
    for filename in os.listdir(labels_folder_path):
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(labels_folder_path, filename)
        image_path = os.path.join(image_folder_path, filename.replace('.txt', '.png'))
        img = Image.open(image_path)
        width, height = img.size
        
        print(f"Image: {image_path} Size: {width}x{height}")
        
        with open(file_path, 'r') as f:
            labels = f.readlines()
        
        with open(file_path, 'w') as f:
            print(f"Label: {file_path}")
            
            for label in labels:
                label = label.strip().split(' ')
                cls = label[0]
                
                if cls in classes:
                    cls_id = classes[cls]
                    x_min, y_min, x_max, y_max = float(label[4]), float(label[5]), float(label[6]), float(label[7])
                    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                    w, h = x_max - x_min, y_max - y_min
                    x, y, w, h = x_center / width, y_center / height, w / width, h / height
                    
                    if x > 1 or y > 1 or w > 1 or h > 1:
                        print(f"Over the limit file: {file_path} data: {cls_id} {x} {y} {w} {h}")
                    
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert dataset labels to YOLOv5 format.')
parser.add_argument('--data_folder', type=str, default='kit', help='Path to the root dataset folder that contains train, val, test folders.')
args = parser.parse_args()

# Define folder paths
data_folder = args.data_folder
train_labels_folder = os.path.join(data_folder, 'train', 'labels')
val_labels_folder = os.path.join(data_folder, 'val', 'labels')
test_labels_folder = os.path.join(data_folder, 'test', 'labels')
train_images_folder = os.path.join(data_folder, 'train', 'images')
val_images_folder = os.path.join(data_folder, 'val', 'images')
test_images_folder = os.path.join(data_folder, 'test', 'images')

# Convert train labels
convert_folder_kitti_to_yolo_v5(train_labels_folder, train_images_folder)

# Convert val labels
convert_folder_kitti_to_yolo_v5(val_labels_folder, val_images_folder)

# Convert test labels
convert_folder_kitti_to_yolo_v5(test_labels_folder, test_images_folder)































'''def convert_folder_kitti_to_yolo_v5(labels_folder_path, output_folder_path, image_folder_path):
    classes = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, "Person_sitting": 4, "Cyclist": 5, "Tram": 6, "Misc": 7}
    
    for filename in os.listdir(labels_folder_path):
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(labels_folder_path, filename)
        image_path = os.path.join(image_folder_path, filename.replace('.txt', '.png'))
        img = Image.open(image_path)
        width, height = img.size
        
        print(f"Image: {image_path} Size: {width}x{height}")
        
        with open(file_path, 'r') as f:
            labels = f.readlines()
        
        output_path = os.path.join(output_folder_path, filename)
        
        with open(output_path, 'w') as f:
            print(f"Label: {output_path}")
            
            for label in labels:
                label = label.strip().split(' ')
                cls = label[0]
                
                if cls in classes:
                    cls_id = classes[cls]
                    x_min, y_min, x_max, y_max = float(label[4]), float(label[5]), float(label[6]), float(label[7])
                    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                    w, h = x_max - x_min, y_max - y_min
                    x, y, w, h = x_center / width, y_center / height, w / width, h / height
                    
                    if x > 1 or y > 1 or w > 1 or h > 1:
                        print(f"Over the limit file: {file_path} data: {cls_id} {x} {y} {w} {h}")
                    
                    f.write(f"{cls_id} {x} {y} {w} {h}\n")

# Define folder paths
train_out_label_dir = 'data/KITTI/old_labels/train/labels'
test_out_label_dir = 'data/KITTI/old_labels/test/labels'
val_out_label_dir = 'data/KITTI/old_labels/validation/labels'

yolo_train_labels = 'data/KITTI/train/labels/'
yolo_test_labels = 'data/KITTI/test/labels/'
yolo_val_labels = 'data/KITTI/validation/labels/'

yolo_train_images = 'data/KITTI/train/images/'
yolo_test_images = 'data/KITTI/test/images/'
yolo_val_images = 'data/KITTI/validation/images/'

# Create output directories if they don't exist
os.makedirs(yolo_train_labels, exist_ok=True)
os.makedirs(yolo_test_labels, exist_ok=True)
os.makedirs(yolo_val_labels, exist_ok=True)

# Convert folders
convert_folder_kitti_to_yolo_v5(train_out_label_dir, yolo_train_labels, yolo_train_images)
convert_folder_kitti_to_yolo_v5(test_out_label_dir, yolo_test_labels, yolo_test_images)
convert_folder_kitti_to_yolo_v5(val_out_label_dir, yolo_val_labels, yolo_val_images)
'''