import cv2
import os
import argparse

def convert_video_to_frames(video_path, output_folder):

    video = cv2.VideoCapture(video_path)
    images_folder = os.path.join(output_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)

    frame_count = 0

    # Read frames from the video and save them as images
    while True:
        # Read the next frame
        success, frame = video.read()

        # Break the loop if there are no more frames
        if not success:
            break

        frame_path = os.path.join(images_folder, f"frame{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    video.release()
    print(f"Frames saved to {images_folder}")


parser = argparse.ArgumentParser(description='Convert video to frames.')
parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
parser.add_argument('--output_folder', type=str, default='frames', help='Path to the output folder to save the frames')
args = parser.parse_args()

convert_video_to_frames(args.video_path, args.output_folder)
