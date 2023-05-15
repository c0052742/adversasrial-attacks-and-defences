import cv2

video_path = 'H:/london_cut.mp4'
frames_dir = 'frames_for_attacks'

video = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_path = f"{frames_dir}/frame{frame_count:04d}.png"
    cv2.imwrite(frame_path, frame)
    frame_count += 1

video.release()


"""
def create_adversarial_video(frames_dir, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24  # Set the desired frames per second for the output video
    frame_size = (1248, 416)  # Set the desired frame size for the output video

    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

    for frame_number in range(frame_count):
        adversarial_frame_path = f"{frames_dir}/frame{frame_number:04d}.png"
        adversarial_frame = cv2.imread(adversarial_frame_path)
        out.write(adversarial_frame)

    out.release()

fgm_video_path = 'adversarial_video/fgm'
pgd_video_path = 'adversarial_video/pgd'
dpt_video_path = 'adversarial_video/dpt'
if not os.path.exists(fgm_video_path):
        os.makedirs(fgm_video_path, exist_ok=True)
if not os.path.exists(pgd_video_path):
    os.makedirs(pgd_video_path, exist_ok=True)
if not os.path.exists(dpt_video_path):
    os.makedirs(dpt_video_path, exist_ok=True)


create_adversarial_video(fgm_frames_dir, fgm_video_path)
create_adversarial_video(pgd_frames_dir, pgd_video_path)
create_adversarial_video(dpt_frames_dir, dpt_video_path)"""