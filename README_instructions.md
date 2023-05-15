# Adversarial attacks and Defences on Autonomous Vehicles

This will show instructions on how to train, create adversarial attacks and defences, and how to evaluate those attacks and defences. 
<ol>
<li> After the repository is cloned, run `./install.sh`, this will install all the dependencies in this project. <em>If there is an error when running AdversarialPatchPytorch, check ART_bug.txt.</em></li>
<li> To access the dataset used in this project, dowload the KITTI dataset from this link https://www.cvlibs.net/datasets/KITTIti/eval_object.php?obj_benchmark=2d and download `Download left color images of object data set (12 GB)`, afterwards upload it to intended directory</li>
<li> Once the images and labels are downloaded, run `split.py` to split the dataset into train,test,val folders and  `convert.py` to convert the labels to YOLOv5 format.<em>If the image shape is not universal, run `image_resize.py` before `convert.py`</em></li>
<li> To train the model follow the instructions in `ultralytics/yolov5` GitHub page</li>
<li> To get the frames from a specific video, run `video_to_frames.py`, to get the labels for the object in the video run `labels_for_frames.py`</li>
<li> To generate the adversarial attacks run `ART_attack.py`, adjust the parameters for specific results, the attacks implemented are Fast Gradient Method, Projected Gradient Descent and Adversarial Patch</li>
<li> To create new training data for `Adversarial Training` run `combine_for_AT.py` and then train the model like mentioned in step 4.</li>
<li> To generate images with preprocessing defences: feature squeezing and spatial smoothing, run `ART_feature_squeezing.py` and `ART_spatial_squeezing.py`</li>
<li> To evaluate the methdods, use the `ultralytics/yolov5` `val.py`, by creating a new .yaml file with the `val:` parameter as the folder to be evaluated against the default or adversarialy_trained model</li>
<li> To create a video from single image frames, run `frames_to_video.py`</li>
</ol>

# Example run
----------Preproccesing----------
```bash
./install.py
```

```bash
python split.py --train_img_dir data/KITTI/data_object_image_2/training/image_2 --train_label_dir data/KITTI/data_object_image_2/data_object_label_2/training/label_2 --train_out_img_dir KITTI/train/images --train_out_label_dir KITTI/train/labels --val_out_img_dir KITTI/val/images --val_out_label_dir KITTI/val/labels --test_out_img_dir KITTI/test/images --test_out_label_dir KITTI/test/labels --val_frac 0.1 --test_frac 0.2
```
```bash
python resize_images.py --input_folder data/KITTI/train/images --target_size 1245,375
python resize_images.py --input_folder data/KITTI/test/images --target_size 1245,375
python resize_images.py --input_folder data/KITTI/val/images --target_size 1245,375
```
```bash
python convert_labels.py --data_folder data/KITTI
```

----------Model training----------
```bash
python train.py --data KITTI.yaml --batch -1 --img 1240 --weights yolov5n.pt --hyp hyp.scratch-med.yaml --cfg yolov5n.yaml --rectangle --epochs 300 --name KITTI_med
python train.py --data KITTI.yaml --batch -1 --img 1240 --weights KITTI_med.pt --hyp KITTI_med_hyp.yaml  --rectangle --epochs 100 --evolve
python train.py --data KITTI.yaml --batch -1 --img 1240 --weights yolov5n.pt --hyp KITTI_med_evolve_hyp.yaml --cfg yolov5n.yaml --rectangle --epochs 300 --name KITTI_med_evolve
 ```
----------Creating the attacks and defences----------
```bash
python video_to_frames.py --video_path input_video.mp4 --output_folder frames
python labels_for_frames.py --images_folder frames --model KITTI_med_evolve.pt
```

```bash
python ART_attack.py --test_image_folder data/KITTI/test --output_folder_fgm adv_attack/fgm --output_folder_pgd adv_attack/pgd --output_folder_ap adv_attack/ap --model KITTI_med_evolve.pt
python combine_for_AT.py --input_folders adv_attack/fgm adv_attack/pgd adv_attack/ap --combined_folder combined_adv_attack --train_folder combined_adv_attack/train --val_folder combined_adv_attack/val --train_ratio 0.8
python ART_feature_squeezing.py --base_FS_image_folder adv_attack --base_output_folder adv_defense/FS --output_prefixes fgm,pgd,ap --bit_depth 7 --default_image_folder data/KITTI/test --model KITTI_med_evolve.pt
python ART_spatial_smoothing.py --base_SS_image_folder adv_attack --base_output_folder adv_defense/SS --output_prefixes fgm,pgd,ap --window_size 5 --default_image_folder data/KITTI/test --model KITTI_med_evolve.pt
```

----------Creating the attacks and defences for video frames----------

```bash
python ART_attack.py --test_image_folder frames --output_folder_fgm adv_attack_video/fgm --output_folder_pgd adv_attack_video/pgd --output_folder_ap adv_attack_video/ap --model KITTI_med_evolve.pt
python ART_feature_squeezing.py --base_FS_image_folder adv_attack_video --base_output_folder adv_defense_video/FS --output_prefixes fgm,pgd,ap --bit_depth 7 --default_image_folder frames --model KITTI_med_evolve.pt
python ART_spatial_smoothing.py --base_SS_image_folder adv_attack_video --base_output_folder adv_defense_video/SS --output_prefixes fgm,pgd,ap --window_size 5 --default_image_folder frames --model KITTI_med_evolve.pt
```


----------Evaluation----------

