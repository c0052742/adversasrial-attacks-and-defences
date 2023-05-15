import os
import random
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Train, Validation, and Test Data')
    parser.add_argument('--train_img_dir', type=str, default='data/KITTI/data_object_image_2/training/image_2', help='Path to the original training images directory')
    parser.add_argument('--train_label_dir', type=str, default='data/KITTI/data_object_image_2/data_object_label_2/training/label_2', help='Path to the original training labels directory')
    parser.add_argument('--train_out_img_dir', type=str, default='kit/train/images', help='Path to the output training images directory')
    parser.add_argument('--train_out_label_dir', type=str, default='kit/train/labels', help='Path to the output training labels directory')
    parser.add_argument('--val_out_img_dir', type=str, default='kit/val/images', help='Path to the output validation images directory')
    parser.add_argument('--val_out_label_dir', type=str, default='kit/val/labels', help='Path to the output validation labels directory')
    parser.add_argument('--test_out_img_dir', type=str, default='kit/test/images', help='Path to the output test images directory')
    parser.add_argument('--test_out_label_dir', type=str, default='kit/test/labels', help='Path to the output test labels directory')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of the data to use for validation')
    parser.add_argument('--test_frac', type=float, default=0.2, help='Fraction of the data to use for testing')
    return parser.parse_args()

def copy_file(src_file_path, dst_file_path):
    if os.path.exists(src_file_path):
        shutil.copy(src_file_path, dst_file_path)
    else:
        print(f"File not found: {src_file_path}")
        return False
    return True

def main():
    args = parse_args()

    os.makedirs(args.train_out_img_dir, exist_ok=True)
    os.makedirs(args.train_out_label_dir, exist_ok=True)
    os.makedirs(args.val_out_img_dir, exist_ok=True)
    os.makedirs(args.val_out_label_dir, exist_ok=True)
    os.makedirs(args.test_out_img_dir, exist_ok=True)
    os.makedirs(args.test_out_label_dir, exist_ok=True)

    img_files = os.listdir(args.train_img_dir)
    random.shuffle(img_files)

    num_val = int(len(img_files) * args.val_frac)
    num_test = int(len(img_files) * args.test_frac)

    for i, img_file in enumerate(img_files[num_val+num_test:]):
        src_img_path = os.path.join(args.train_img_dir, img_file)
        dst_img_path = os.path.join(args.train_out_img_dir, img_file)
        copy_file(src_img_path, dst_img_path)

        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label_path = os.path.join(args.train_label_dir, label_file)
        dst_label_path = os.path.join(args.train_out_label_dir, label_file)
        copy_file(src_label_path, dst_label_path)

        print(f'Copied training example {i+1}/{len(img_files)-num_val-num_test}')

    for i, img_file in enumerate(img_files[num_test:num_val+num_test]):
        src_img_path = os.path.join(args.train_img_dir, img_file)
        dst_img_path = os.path.join(args.val_out_img_dir, img_file)
        copy_file(src_img_path, dst_img_path)

        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label_path = os.path.join(args.train_label_dir, label_file)
        dst_label_path = os.path.join(args.val_out_label_dir, label_file)
        copy_file(src_label_path, dst_label_path)

        print(f'Copied validation example {i+1}/{num_val}')

    for i, img_file in enumerate(img_files[:num_test]):
        src_img_path = os.path.join(args.train_img_dir, img_file)
        dst_img_path = os.path.join(args.test_out_img_dir, img_file)
        copy_file(src_img_path, dst_img_path)

        label_file = img_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label_path = os.path.join(args.train_label_dir, label_file)
        dst_label_path = os.path.join(args.test_out_label_dir, label_file)
        copy_file(src_label_path, dst_label_path)

        print(f'Copied test example {i+1}/{num_test}')

if __name__ == "__main__":
    main()
