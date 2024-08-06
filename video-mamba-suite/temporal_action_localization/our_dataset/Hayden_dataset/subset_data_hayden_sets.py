"""
This script processes tennis game data from multiple sources (a txt file and a JSON file)

"""


import json
import cv2
import os

def truncate_video(input_video_path, output_video_path, start_frame, end_frame):
    # Open the video file
    print(f'start frame : {start_frame} , end frame {end_frame}')
    cap = cv2.VideoCapture(input_video_path)
    print(output_video_path)

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

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= current_frame < end_frame:
            out.write(frame)

        current_frame += 1
        #print(current_frame)

        # Stop the loop if the end_frame is reached
        if current_frame >= end_frame:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f'video saved in {output_video_path}')

Annotations_labels = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
transformed_data = {"database": {}}
frame_rate = 25
output_videos_path = 'videos/videos_truncated_hayden_set'


# print(tennis_data)
txt_paths = ['labels/V006.txt', 'labels/V007.txt', 'labels/V008.txt', 'labels/V009.txt',
             'labels/V010.txt']



for txt_path in txt_paths:
    # Get video_name
    video_name = txt_path.split('labels/')[1].split('.txt')[0]
    video_path = 'videos/raw_videos/'+video_name+'.mp4'
    # open the JSON file
    with open(video_name + '.json', 'r') as file:
        frames_data = json.load(file)
    # open the txt_file
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    annotations = {}
    for line in lines:
        frame, label = line.strip().split()
        frame = int(frame)
        annotations[frame] = label

    frames_categories = frames_data.get('classes', {}).get('SPLITS', [])

    #print(frames_categories)
    for frame_category in frames_categories:
        split_type = frame_category['custom']['Type']
        #print(split_type)
        split_start = frame_category['start']
        split_end = frame_category['end']
        #print(f'split_start : {split_start}  ; split_end : {split_end}' )

        total_split_frames = split_end - split_start


        segment_index = 0
        end_frame = 0


        while end_frame != split_end:
            if segment_index == 0 :
                start_frame = split_start
            else :
                start_frame = end_frame + 1
            end_frame = int(min((segment_index + 1) * 5250 + split_start, split_end))

            end_frame_label = annotations[end_frame]

            while end_frame < split_end and annotations.get(end_frame + 1) == end_frame_label:
                end_frame += 1
            segment_name = f"{video_name}_segment_{segment_index + 1}_split_type_{split_type}"
            output_video_path = os.path.join(output_videos_path,segment_name+'.mp4')

            truncate_video(video_path,output_video_path,start_frame,end_frame)
            #print(split_type)
            file_data = {
                "subset": split_type,
                "duration": round((end_frame - start_frame) / frame_rate, 2),
                "fps": frame_rate,
                "annotations": []
            }

            #print(file_data)



            chunk_start = start_frame
            current_label = annotations[chunk_start]

            #print(f'start_frame {start_frame} , end_frame {end_frame}')

            for i in range(start_frame, end_frame):
                frame, label = i, annotations[i]
                if label != current_label:
                    # End of the current segment
                    chunk_end = i - 1
                    file_data["annotations"].append({
                        "label": current_label,
                        "segment": [round((chunk_start / frame_rate) - (start_frame / frame_rate), 2),
                                    round((chunk_end / frame_rate) - (start_frame / frame_rate), 2)],
                        "segment(frames)": [chunk_start - start_frame, chunk_end - start_frame],
                        "label_id": Annotations_labels.index(current_label)
                    })
                    # Start a new chunk
                    chunk_start = frame
                    current_label = label

            file_data["annotations"].append({
                "label": current_label,
                "segment": [round((chunk_start / frame_rate) - (start_frame / frame_rate), 2),
                            round((end_frame / frame_rate) - (start_frame / frame_rate), 2)],
                "segment(frames)": [chunk_start - start_frame, end_frame - start_frame],
                "label_id": Annotations_labels.index(current_label)
            })

            segment_index += 1

            #print(file_data)

            transformed_data["database"][segment_name] = file_data

output_file_path = 'Tennis_hayden.json'
#print(transformed_data)




with open(output_file_path, 'w') as output_file:
    json.dump(transformed_data, output_file, indent=4)
















