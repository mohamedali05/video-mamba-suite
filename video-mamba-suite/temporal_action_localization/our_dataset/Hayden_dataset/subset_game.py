"""
This script processes tennis game data from multiple sources (a txt file and a JSON file)
and processes them to be all in one single JSON File suited for processing in video_mamba_suite.
"""

import json
import random
import os
# Load the main JSON file


# Load the file with the game frames information

Annotations_labels = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
transformed_data = {"database": {}}

frame_rate = 25


game_subsets = {}


json_dir = 'Json_files'
#print(tennis_data)
txt_paths = ['labels/V006.txt','labels/V007.txt','labels/V008.txt','labels/V009.txt' ,'labels/V010.txt' ]
for txt_path in txt_paths :
    #Get video_name
    video_name = txt_path.split('labels/')[1].split('.txt')[0]
    video_path = os.path.join(json_dir, video_name+'.json')
    with open(video_path, 'r') as file:
        frames_data = json.load(file)

    game_frames = frames_data.get('classes', {}).get('Game', [])
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # Process the lines to extract frame numbers and labels
    annotations = {}
    for line in lines:
        frame, label = line.strip().split()
        frame = int(frame)
        annotations[frame] = label



    for game in game_frames:
        game_name = video_name+'_game_'+game.get('name')
        start_frame = game.get('start')
        end_frame = game.get('end')
        list_annotations = []
        file_data = {
            "subset": 'train' if random.random() < 0.7 else 'test',
            "duration": round((end_frame-start_frame)/frame_rate , 2) ,
            "fps": frame_rate,
            "annotations": []
        }

        current_label = annotations[start_frame]
        segment_start = start_frame
        #print(f'start_frame = {start_frame} , end_frame = {end_frame} , game = {game_name}')

        for i in range(start_frame , end_frame) :
            frame, label = i, annotations[i]
            if label != current_label:
                # End of the current segment
                segment_end = i-1
                file_data["annotations"].append({
                    "label": current_label,
                    "segment": [round((segment_start / frame_rate)-(start_frame/frame_rate) ,2) , round((segment_end/frame_rate)-(start_frame/frame_rate) , 2) ],
                    "segment(frames)": [segment_start - start_frame, segment_end -start_frame],
                    "label_id": Annotations_labels.index(current_label)
                })
                # Start a new segment
                segment_start = frame
                current_label = label

        # Add the last segment
        file_data["annotations"].append({
            "label": current_label,
            "segment": [round((segment_start/frame_rate)-(start_frame/frame_rate) , 2),round((end_frame/frame_rate)-(start_frame/frame_rate) , 2) ],
            "segment(frames)": [segment_start - start_frame, end_frame - start_frame],
            "label_id": Annotations_labels.index(current_label)
        })



        transformed_data["database"][game_name] = file_data
        output_file_path = 'Tennis_new.json'

        with open(output_file_path, 'w') as output_file:
            json.dump(transformed_data, output_file, indent=4)







    
