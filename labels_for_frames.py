import os
import torch
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder', type=str, required=True, help='Directory for the .../images to be processed .')
parser.add_argument('--model', type=str, required=True, help='Path to the trained YOLOv5 model.')
args = parser.parse_args()
def normalize_bbox(x1, y1, x2, y2, img_width, img_height):
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model)

# Set the path to the folder containing images
main_folder = args.images_folder
image_folder = os.path.join(main_folder, 'images')
image_filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create a directory for labels if it doesn't exist
label_folder = os.path.join(main_folder, 'labels')
os.makedirs(label_folder, exist_ok=True)

# Process each image in the folder
for image_filename in image_filenames:
    # Load an image
    image_path = os.path.join(image_folder, image_filename)
    image = Image.open(image_path)

    # Perform inference
    results = model(image)

    # Prepare the label file path
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(label_folder, label_filename)

    # Extract bounding boxes, object classes, and confidence scores, and save them to the label file
    img_width, img_height = image.size
    with open(label_path, 'w') as label_file:
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x_center, y_center, width, height = normalize_bbox(x1, y1, x2, y2, img_width, img_height)
            confidence = float(conf)
            class_id = int(cls)

            label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Saved labels for {image_filename} in {label_path}")
