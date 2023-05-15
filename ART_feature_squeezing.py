import os
import numpy as np
from PIL import Image
import torch
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.defences.preprocessor import FeatureSqueezing
from utils.loss import ComputeLoss
import argparse
import shutil

class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.hyp = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "fl_gamma": 0.0,
        }
        self.compute_loss = ComputeLoss(self.model.model.model)
    def forward(self, x, targets=None):
        if self.training:
            targets = targets.to(self.model.model.device)
            outputs = self.model.model.model(x)
            loss, loss_components = self.compute_loss(outputs, targets)

            loss_components_dict = {"loss_total": loss,
                                    "loss_box": loss_components[0],
                                    "loss_obj": loss_components[1],
                                    "loss_cls": loss_components[2]}
            return loss_components_dict
        else:
            return self.model(x)

config = {"attack_losses": ["loss_total", "loss_box", "loss_obj", "loss_cls"],}

parser = argparse.ArgumentParser()
parser.add_argument('--base_FS_image_folder', type=str, default=None, help='path to the base folder containing images for featuree squeezing for the different attacks')
parser.add_argument('--base_output_folder', type=str, required=True, help='base path to the folder where output images will be saved')
parser.add_argument('--output_prefixes', type=str, default='ap,fgm,pgd', help='prefixes for output and test directory folders, separated by commas')
parser.add_argument('--bit_depth',type=int,default=7,help='bit_depth value for feature squeezing')
parser.add_argument('--default_image_folder',type=str,default=None,help='folder for default images without the adversarial attacks.')
parser.add_argument('--model', type=str, required=True, help='Path to the trained YOLOv5 model.')
args = parser.parse_args()

model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model, force_reload=True)
model.eval()
model = Yolo(model)
feature_squeezing = FeatureSqueezing(clip_values=[0, 255], bit_depth=7)
estimator = PyTorchYolo(model=model, device_type="cuda", input_shape=(3, 1248, 416), clip_values=(0, 255), attack_losses=config["attack_losses"], preprocessing_defences=[feature_squeezing])

if args.base_FS_image_folder is not None:
    folders = args.output_prefixes.split(',')
    base_FS_image_folder = args.base_FS_image_folder
bit_depth = args.bit_depth
base_output_folder = args.base_output_folder

bit_depth_ratio = 2 ** (8 - bit_depth)

def copy_labels(original_folder, new_folder, filename):
    label_filename = filename.replace('.png', '.txt')
    original_label_path = os.path.join(original_folder.replace('images', 'labels'), label_filename)
    new_label_path = os.path.join(new_folder, label_filename)
    os.makedirs(new_folder, exist_ok=True)
    shutil.copy2(original_label_path, new_label_path)

def save_squeezed_image(image_path, output_folder, estimator, bit_depth_ratio):
    image = np.asarray(Image.open(image_path).resize((1248, 416)))  
    img_reshape = image.transpose((2, 0, 1))
    im = np.stack([img_reshape], axis=0).astype(np.float32)
    x = im.copy()

    prediction = estimator.predict(x)

    squeezed_image = x[0].transpose((1, 2, 0)) 
    squeezed_image = np.clip(squeezed_image * bit_depth_ratio, 0, 255)
    output_image = Image.fromarray(squeezed_image.astype(np.uint8))
    os.makedirs(output_folder, exist_ok=True)
    output_image.save(os.path.join(output_folder, os.path.basename(image_path)))
    

if args.base_FS_image_folder is not None:
    for folder in folders:
        test_image_folder = os.path.join(base_FS_image_folder, folder, "images")
        filenames = os.listdir(test_image_folder)

        for filename in filenames:
            image_path = os.path.join(test_image_folder, filename)
            image = np.asarray(Image.open(image_path).resize((1248, 416)))  
            img_reshape = image.transpose((2, 0, 1))
            im = np.stack([img_reshape], axis=0).astype(np.float32)
            x = im.copy()

            prediction = estimator.predict(x)

            # Save the processed image
            squeezed_image = x[0].transpose((1, 2, 0)) 
            squeezed_image = np.clip(squeezed_image * bit_depth_ratio, 0, 255)
            output_image = Image.fromarray(squeezed_image.astype(np.uint8))
            output_folder = os.path.join(base_output_folder, "FS_" + folder, "images")
            os.makedirs(output_folder, exist_ok=True)
            output_image.save(os.path.join(output_folder, filename))
            copy_labels(test_image_folder, output_folder.replace('images', 'labels'), filename)


if args.default_image_folder is not None:
    default_test_image_folder = args.default_image_folder
    default_output_folder = os.path.join(base_output_folder, "default", "images")
    default_filenames = os.listdir(default_test_image_folder)

    for filename in default_filenames:
        image_path = os.path.join(default_test_image_folder, filename)
        save_squeezed_image(image_path, default_output_folder, estimator, bit_depth_ratio)
        copy_labels(test_image_folder, output_folder.replace('images', 'labels'), filename)