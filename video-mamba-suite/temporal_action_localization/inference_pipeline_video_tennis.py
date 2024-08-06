"""
This script processes video annotations and generates output files, including JSON files and annotated videos.
It is designed for a specific use case involving a video dataset related to tennis matches. The script performs the following key tasks:

1. **Configuration Loading:** Reads configuration settings from a text file to set up model parameters and paths.
2. **Model Setup:** Initializes and loads a pre-trained model for temporal action localization in videos.
3. **Feature and Video Processing:**
   - Extracts features and video metadata, such as frame rate and duration.
   - Applies the model to the extracted features to obtain action segments and scores.
4. **Annotation and Video Creation:**
   - Generates JSON files containing the annotations with action labels and segments.
   - Creates annotated videos with labels overlaid in English and/or French, optionally including frame numbers.

Command-line Arguments:
- `--create_json`: If specified, the script generates a JSON file with the detected annotations.
- `--create_english_video`: If specified, the script generates an annotated video in English.
- `--create_french_video`: If specified, the script generates an annotated video in French.
- `--add_frame_number`: If specified, the script adds the frame number to the annotated video output.
"""




# python imports



import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import json
import mmengine
import io
import hashlib
import numpy as np
from libs.modeling import make_meta_arch
import cv2
import ast


def load_config_from_txt(config_file):
    """
       Loads a configuration from a text file using Python's literal evaluation.

       Args:
           config_file (str): Path to the configuration file.

       Returns:
           dict: Configuration dictionary.
    """
    with open(config_file, 'r') as file:
        config_content = file.read()
        config = ast.literal_eval(config_content)
    return config


def get_frame_rate(file_video):
    """
        Retrieves the frame rate of a video.

        Args:
            file_video (str): Path to the video file.

        Returns:
            float: Frame rate of the video.
    """
    video_capture = cv2.VideoCapture(file_video)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    return frame_rate


def get_total_frames(file_video):
    """
        Retrieves the total number of frames in a video.
        Args:
            file_video (str): Path to the video file.
        Returns:
            int: Total number of frames in the video.
    """
    video_capture = cv2.VideoCapture(file_video)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    return total_frames


def get_duration(file_video):
    """
        Calculates the duration of a video in seconds.
        Args:
            file_video (str): Path to the video file.

        Returns:
            float: Duration of the video in seconds.
    """
    video_capture = cv2.VideoCapture(file_video)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    duration_in_seconds = total_frames / frame_rate
    return duration_in_seconds


def add_text(frame, text, frame_number, font_scale , add_frame_number):
    """
        Adds text to a video frame. Can include annotations and frame numbers.
        Args:
            frame (ndarray): The video frame.
            text (str): The text to add.
            frame_number (int): The current frame number.
            font_scale (float): Scale factor for the text size.
            add_frame_number (bool): Whether to include the frame number in the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    thickness = 2
    margin = 10
    line_type = cv2.LINE_AA

    colon_index = text.find(':')
    if colon_index != -1:
        first_part = text[:colon_index + 1]
        remaining_parts = text[colon_index + 1:].strip()
        lines = [first_part] + remaining_parts.split(':')
    else:
        lines = [text]

    start_x = frame.shape[1]
    start_y = margin

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = start_x - text_size[0] - margin
        text_y = start_y + text_size[1] + margin

        cv2.rectangle(frame, (text_x - margin, text_y - text_size[1] - margin),
                      (text_x + text_size[0] + margin, text_y + margin), bg_color, -1)
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

        start_y += text_size[1] + margin + 10


    if add_frame_number :
        frame_text = f'Frame: {frame_number}'
        text_size = cv2.getTextSize(frame_text, font, font_scale, thickness)[0]
        text_x = start_x - text_size[0] - margin
        text_y = start_y + text_size[1] + margin

        cv2.rectangle(frame, (text_x - margin, text_y - text_size[1] - margin),
                      (text_x + text_size[0] + margin, text_y + margin), bg_color, -1)

        cv2.putText(frame, frame_text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)



def main(create_json, create_english_video = None, create_french_video = None , add_frame_number = None) :
    #model_path = './ckpt_tennis/2024-07-08_16-51-19_best_params_3_classes' #path to the model
    features_path = os.path.join('our_dataset','test_set', 'game_features')
    videos_path = os.path.join('our_dataset','test_set','game_videos')
    model_path = './ckpt_tennis/2024-06-25_13-27-21_3_classes_no_other_between_2_exch'  #path to the model
    config_path = os.path.join(model_path, 'config.txt')
    ckpt_path = os.path.join(model_path,'model_best.pth.tar')
    cfg = load_config_from_txt(config_path)

    output_dir = os.path.join("videos_annotated_inference")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir,  'annotations.json')


    #print(cfg)




    feat_stride = cfg['dataset']['feat_stride']
    feat_num_frames = cfg['dataset']['num_frames']


    #Annotations is the list that contains the label names written in the JSON file
    if cfg['dataset']['num_classes'] == 3 :

        Annotations = ["PHASE DE NON-JEU",
                       "SERVICE",
          "ECHANGE"]
    else :

        Annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
        '''
        Annotations = ["OTH",
                       "SERVICE",
                       "Service fautif du joueur éloigné de la caméra",
                       "Service avec let (rencontre du filet) du joueur éloigné de la caméra",
                       "Service réussi du joueur proche de la caméra",
                       "Service fautif du joueur proche de la caméra",
                       "Service avec let (rencontre du filet) du joueur proche de la caméra",
                       "ECHANGE ",
                       "ECHANGE ",
                       "ECHANGE ",
                       "ECHANGE "]
        '''



    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    checkpoint = torch.load(ckpt_path,
                            map_location=lambda storage, loc: storage.cuda(
                                cfg['devices'][0]))

    model.load_state_dict(checkpoint['state_dict'] )


    features = []
    for file in os.listdir(features_path) :
        #print(file)
        if not file.endswith('.npy') or 'Zone.Identifier' in file:
            continue


        video_id = file.replace('.npy' , '')
        duration = get_duration(os.path.join(videos_path, video_id + '.mp4'))
        fps = get_frame_rate(os.path.join(videos_path, video_id + '.mp4'))
        data = io.BytesIO(mmengine.get(os.path.join(features_path , file)))
        feats = np.load(data, allow_pickle=True).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        data = [{'video_id' : video_id , 'feats' : feats , 'fps' : fps  , 'duration' : duration , 'feat_stride' : feat_stride , 'feat_num_frames' : feat_num_frames}]
        features.append(data)

    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)


    dict_annotations = {}
    for iter_idx, video_list in enumerate(features, 0):
        # forward the model (wo. grad)
        print(f'iter_idx {iter_idx}')
        with torch.no_grad():

            output = model(video_list)
            num_vids = len(output)


            for vid_idx in range(num_vids):
                print(f'vid_idx {vid_idx}')
                print(f'num_vids {num_vids}')

                results_per_video = []
                file_name = output[vid_idx]['video_id']
                print(file_name)
                video_file = os.path.join(videos_path, file_name+'.mp4')
                subset, duration, fps, total_frames = "test", get_duration(video_file), get_frame_rate(
                    video_file), get_total_frames(video_file)
                dict_annotations[file_name] = {'subset': subset, 'duration': duration, 'fps': fps, 'annotations': []}
                segments = output[vid_idx]['segments']
                scores = output[vid_idx]['scores']
                labels = output[vid_idx]['labels']
                for seg, label, score in zip(segments, labels, scores):
                    result = {
                        "label":  Annotations[label.item()],
                        "segment": [seg[0].item(), seg[1].item()],  # Convert segment tensor to list
                        "score": score.item()
                    }
                    results_per_video.append(result)
                cap = cv2.VideoCapture(video_file)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                Annotations_translated = {"PHASE DE NON-JEU": "OTHER", 'ECHANGE': 'EXCHANGE', 'SERVICE': 'SERVICE'}
                out_english = None
                out_french = None

                if create_english_video:
                    out_english = cv2.VideoWriter(
                        os.path.join(output_dir, "english_annotated_" +file_name+'.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                if create_french_video:
                    out_french = cv2.VideoWriter(
                        os.path.join(output_dir, "french_annotated_" +file_name+'.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                segments = [(int(item['segment'][0] * fps), int(item['segment'][1] * fps), item['label']) for item in
                            results_per_video]
                current_frame = 0
                previous_label = ''
                start_frame_label = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    for start_frame, end_frame, label in segments:
                        if start_frame <= current_frame < end_frame:
                            if create_english_video:
                                add_text(frame, Annotations_translated[label], current_frame, 1.5, add_frame_number)
                                out_english.write(frame)

                            if create_french_video:
                                add_text(frame, label, current_frame, 1.5, add_frame_number)
                                out_french.write(frame)
                            if create_json and previous_label != label and previous_label != '':
                                dict_annotations[file_name]['annotations'].append({'label': previous_label,
                                                                                   'segment': [
                                                                                   round(start_frame_label / fps,
                                                                                         2),
                                                                                   round((current_frame - 1) / fps,
                                                                                         2)],
                                                                               'segment(frames)': [
                                                                                   start_frame_label,
                                                                                   current_frame - 1],
                                                                               'label_id': Annotations.index(
                                                                                       previous_label)})

                                start_frame_label = current_frame
                            previous_label = label
                            break

                    if create_json and current_frame == total_frames - 1:
                        dict_annotations[file_name]['annotations'].append({'label': previous_label,
                                                                           'segment': [
                                                                               round(start_frame_label / fps, 2),
                                                                               round(total_frames / fps, 2)],
                                                                           'segment(frames)': [start_frame_label,
                                                                                               int(total_frames)],
                                                                           'label_id': Annotations.index(previous_label)})

                    current_frame += 1

                if create_english_video:
                    out_english.release()
                    print(f'finishing producing the video : english_annotated_'+file_name+'.mp4')
                if create_french_video:
                    out_french.release()
                    print(f'finishing producing the video : french_annotated_'+file_name+'.mp4')
                cap.release()

    if create_json:
        with open(output_file_path, 'w') as output_file:
            json.dump({'database': dict_annotations}, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video annotations and generate outputs.')
    parser.add_argument('--create_json', action='store_true', help='Create the JSON file.')
    parser.add_argument('--create_english_video', action='store_true', help='Create annotated videos in English.')
    parser.add_argument('--create_french_video', action='store_true', help='Create annotated videos in French.')
    parser.add_argument('--add_frame_number', action='store_true', help='adds frame number to the output of the video')
    args = parser.parse_args()


    main(args.create_json, args.create_english_video, args.create_french_video, args.add_frame_number)






