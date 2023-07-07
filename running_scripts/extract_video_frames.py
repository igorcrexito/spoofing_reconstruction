import cv2
import os


def sample_frames(video_path, num_frames, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the step size based on the number of frames to be sampled
    step_size = max(total_frames // num_frames, 1)

    # Create a directory to save the sampled frames
    os.makedirs(output_dir, exist_ok=True)

    # Read and save the sampled frames
    frame_count = 0
    while True:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Read the current frame
        ret, frame = video.read()
        if not ret:
            break

        # Resize the frame to 256x256x3 dimensions
        resized_frame = cv2.resize(frame, (256, 256))

        # Save the frame as a PNG image
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, resized_frame)

        # Increment the frame count by the step size
        frame_count += step_size

    # Release the video file
    video.release()


if __name__ == '__main__':

    video_folder = '../../video_samples/'
    num_frames = 32  # Number of frames to sample from each video

    video_list = os.listdir(video_folder)

    for video_file in video_list:
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_path = os.path.join(video_folder, video_file)
            sample_frames(video_path=video_path, num_frames=num_frames, output_dir=f'{video_path[:-4]}/')
