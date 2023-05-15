#!/bin/bash
python detect.py --weights 'runs/train/KITTI_med_evolve/weights/best.pt' --source 'adversarial_images/fgm' --name fgm
python detect.py --weights 'runs/train/KITTI_med_evolve/weights/best.pt' --source 'adversarial_images/dpt' --name dpt       
python detect.py --weights 'runs/train/KITTI_med_evolve/weights/best.pt' --source 'adversarial_images/pgd' --name pgd