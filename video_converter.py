import cv2
import os
import argparse

def create_video_from_images(image_folder, output_video_name):
    images = sorted(os.listdir(image_folder))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    video.release()
    print(f"Video created: {output_video_name}")

parser = argparse.ArgumentParser()
parser.add_argument('--video_frames_folder', type=str, required=True, help='A folder that contains the frame images.')
parser.add_argument('--video_name', type=str, default='video', help='Name for the video generated')
args = parser.parse_args()


output_video_name = f"{args.video_name}.mp4"
create_video_from_images(args.video_frames_folder, output_video_name)
