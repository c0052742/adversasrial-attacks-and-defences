from utils.loss import ComputeLoss
import os
import shutil
import numpy as np
from PIL import Image
import torch
import argparse
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent,AdversarialPatchPyTorch


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
            targets = targets.to(self.model.model.device)  # Move targets to the same device as the model
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
parser.add_argument('--test_image_folder', type=str, required=True, help='path to the folder containing test images and labels folders')
parser.add_argument('--output_folder_fgm', type=str, required=True, help='path to the folder where fgm adversarial images will be saved with labels')
parser.add_argument('--output_folder_pgd', type=str, required=True, help='path to the folder where pgd adversarial images will be saved with labels')
parser.add_argument('--output_folder_ap', type=str, required=True, help='path to the folder where ap adversarial images will be saved with labels')
parser.add_argument('--model', type=str, required=True, help='Path to the trained YOLOv5 model.')
args = parser.parse_args()

model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model)
model.eval()
model = Yolo(model)

estimator = PyTorchYolo(
    model=model, device_type="cuda", input_shape=(3, 1248, 416), clip_values=(0, 255), attack_losses=config["attack_losses"],
)

test_image_folder = args.test_image_folder
output_folder_fgm = args.output_folder_fgm
output_folder_pgd = args.output_folder_pgd
output_folder_ap = args.output_folder_ap

image_folder = os.path.join(test_image_folder, 'images')
image_filenames = os.listdir(image_folder)
image_filenames = [os.path.join(image_folder, filename) for filename in image_filenames if filename.endswith('.png')]


label_folder = os.path.join(test_image_folder, 'labels')
output_folder_fgm_labels = os.path.join(output_folder_fgm, 'labels')
output_folder_pgd_labels = os.path.join(output_folder_pgd, 'labels')
output_folder_ap_labels = os.path.join(output_folder_ap, 'labels')

# Loop over images and generate adversarial images and corresponding labels
for filename in image_filenames:
    image_path = filename
    image = np.asarray(Image.open(image_path).resize((1248, 416)))  
    img_reshape = image.transpose((2, 0, 1))
    im = np.stack([img_reshape], axis=0).astype(np.float32)
    x = im.copy()

    fgm = FastGradientMethod(estimator=estimator,
                             eps=17,
                             targeted=False,
                             batch_size=1,
                             summary_writer=False,)

    adversarial_image_fgm = fgm.generate(x=x, y=None)

    pgd = ProjectedGradientDescent(
        estimator=estimator,
        norm=np.inf,  
        eps=25,  
        eps_step=2,  
        random_eps=False, 
        decay=0.7,  
        max_iter=100,  
        targeted=False, 
        batch_size=1,  
        summary_writer=False,  
        verbose=True 
    )
    adversarial_image_pgd = pgd.generate(x=x)
    adversarial_image_fgm = adversarial_image_fgm.squeeze(0).transpose(1, 2, 0)
    adversarial_image_pgd = adversarial_image_pgd.squeeze(0).transpose(1, 2, 0)
    
    target = estimator.predict(x)
    rotation_max=22.5
    scale_min=0.4
    scale_max=1.0
    learning_rate=1.99
    batch_size=1
    max_iter=500
    patch_shape=(3, 640, 640)
    optimizer = 'pgd'


    ap = AdversarialPatchPyTorch(
                rotation_max=rotation_max,
                estimator=estimator,
                scale_min=scale_min,
                scale_max=scale_max,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_iter=max_iter,
                patch_shape=patch_shape,
                verbose=True,
                optimizer=optimizer,
                targeted=False
            )

    patch, patch_mask = ap.generate(x=x,y=target)
    adversarial_image_ap = ap.apply_patch(x, scale=0.25)

    adversarial_image_fgm = np.uint8(adversarial_image_fgm)
    adversarial_image_pgd = np.uint8(adversarial_image_pgd)
    if not os.path.exists(output_folder_fgm):
        os.makedirs(output_folder_fgm, exist_ok=True)
    if not os.path.exists(output_folder_pgd):
        os.makedirs(output_folder_pgd, exist_ok=True)
    if not os.path.exists(output_folder_ap):
        os.makedirs(output_folder_ap, exist_ok=True)
    
    

    adversarial_image_fgm = Image.fromarray(adversarial_image_fgm)
    adversarial_image_pgd = Image.fromarray(adversarial_image_pgd)
    adversarial_image_ap = np.transpose(adversarial_image_ap.squeeze(0), (1, 2, 0)).astype(np.uint8)

    os.makedirs(os.path.join(output_folder_ap, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_fgm, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_pgd, 'images'), exist_ok=True)

    Image.fromarray(adversarial_image_ap).save(os.path.join(output_folder_ap,'images', 'ap_' + os.path.basename(filename)))
    adversarial_image_fgm.save(os.path.join(output_folder_fgm,'images', 'fgm_' + os.path.basename(filename)))
    adversarial_image_pgd.save(os.path.join(output_folder_pgd, 'images','pgd_' + os.path.basename(filename)))

    label_filename = os.path.basename(filename).replace('.png', '.txt')

    if not os.path.exists(output_folder_fgm_labels):
        os.makedirs(output_folder_fgm_labels, exist_ok=True)
    if not os.path.exists(output_folder_pgd_labels):
        os.makedirs(output_folder_pgd_labels, exist_ok=True)
    if not os.path.exists(output_folder_ap_labels):
        os.makedirs(output_folder_ap_labels, exist_ok=True)

    original_label_path = os.path.join(label_folder, label_filename)
    new_fgm_label_path = os.path.join(output_folder_fgm_labels, 'fgm_' + label_filename)
    new_pgd_label_path = os.path.join(output_folder_pgd_labels, 'pgd_' + label_filename)
    new_ap_label_path = os.path.join(output_folder_ap_labels, 'ap_' + label_filename)

    shutil.copy2(original_label_path, new_fgm_label_path)
    shutil.copy2(original_label_path, new_pgd_label_path)
    shutil.copy2(original_label_path, new_ap_label_path)
