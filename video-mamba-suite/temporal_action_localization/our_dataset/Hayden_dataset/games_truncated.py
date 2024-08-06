import json
import os
import cv2


def truncate_video(input_video_path, output_video_path, start_frame, end_frame):
    """
    Truncates a video file to include only the frames between start_frame and end_frame.

    Parameters:
        input_video_path (str): The path to the source video file.
        output_video_path (str): The path where the truncated video will be saved.
        start_frame (int): The frame number to start the truncation.
        end_frame (int): The frame number to end the truncation.

    Returns:
        None. The truncated video is written to output_video_path.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get width and height of video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= current_frame < end_frame:
            out.write(frame)

        current_frame += 1

        # Stop the loop if the end_frame is reached
        if current_frame >= end_frame:
            break

    # Release everything if job is finished
    cap.release()
    out.release()


def main():
    """
    Main function to process multiple videos as specified by JSON metadata.

    Processes each video specified in a hardcoded list, truncating them according
    to frame ranges defined in JSON metadata files associated with each video.
    Outputs the truncated videos to a specified directory.

    Note that you have to define all the paths accordingly so that the script works
    """

    video_paths = ['videos/V006.mp4', 'videos/V007.mp4', 'videos/V008.mp4', 'videos/V009.mp4', 'videos/V010.mp4']
    output_folder = 'games_truncated'

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for video_path in video_paths:
        video_name = video_path.split('.mp4')[0]

        # Load JSON metadata for the video
        json_path = os.path.join('Json_files', video_name + '.json')

        with open(json_path + '.json', 'r') as file:
            json_file = json.load(file)

        # Process each game segment defined in the JSON file
        game_frames = json_file.get('classes', {}).get('Game', [])
        for game in game_frames:
            # Filter to process only certain games based on video and game names

            game_name = video_name + '_game_' + game.get('name')
            start_frame = game.get('start')
            end_frame = game.get('end')

            output_video_path = os.path.join(output_folder, f"{game_name}.mp4")
            truncate_video(video_path, output_video_path, start_frame, end_frame)
            print(f"Truncated video saved to: {output_video_path}")


if __name__ == "__main__":
    main()
