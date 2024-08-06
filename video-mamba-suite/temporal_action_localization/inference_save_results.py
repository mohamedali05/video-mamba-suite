
"""
This script performs inference on a dataset of tennis game videos using a pre-trained action localization model.
It saves all the predictions in JSON Files.

### Key Functionalities:

1. **Configuration and Model Loading:**
   - Loads configuration settings and model parameters from specified paths.
   - Initializes the model for evaluation.

2. **Feature Extraction:**
   - Reads feature files (`.npy`) and associated video metadata (e.g., frame rate, duration) for each video in the dataset.

3. **Model Inference:**
   - Runs the model in evaluation mode to predict action segments, labels, and scores for each video.

4. **Results Saving:**
   - Saves the prediction results for each video in JSON format, including the label, segment times, and confidence score.
"""


from inference_pipeline_video_tennis import load_config_from_txt, get_frame_rate , get_total_frames , get_duration
import os
import torch
import torch.nn as nn
import torch.utils.data
import json
import mmengine
import io
import numpy as np
from libs.modeling import make_meta_arch

def main() :
    features_path = os.path.join('our_dataset','test_set', 'game_features')
    videos_path = os.path.join('our_dataset','test_set','game_videos')
    model_path = './ckpt_tennis/2024-07-08_16-51-19_best_params_3_classes'  # path to the model
    config_path = os.path.join(model_path, 'config.txt')
    ckpt_path = os.path.join(model_path, 'model_best.pth.tar')
    cfg = load_config_from_txt(config_path)

    output_folder_directory = os.path.join('results', 'Results_test_set')


    # print(cfg)

    feat_stride = cfg['dataset']['feat_stride']
    feat_num_frames = cfg['dataset']['num_frames']

    # Annotations is the list that contains the label names written in the JSON file
    if cfg['dataset']['num_classes'] == 3:

        Annotations = ["PHASE DE NON-JEU",
                       "SERVICE",
                       "ECHANGE"]
    else:
        Annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    checkpoint = torch.load(ckpt_path,
                            map_location=lambda storage, loc: storage.cuda(
                                cfg['devices'][0]))

    model.load_state_dict(checkpoint['state_dict'])

    features = []
    for file in os.listdir(features_path):
        # print(file)
        if not file.endswith('.npy') or 'Zone.Identifier' in file:
            continue

        video_id = file.replace('.npy', '')
        duration = get_duration(os.path.join(videos_path, video_id + '.mp4'))
        fps = get_frame_rate(os.path.join(videos_path, video_id + '.mp4'))
        data = io.BytesIO(mmengine.get(os.path.join(features_path, file)))
        feats = np.load(data, allow_pickle=True).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        data = [{'video_id': video_id, 'feats': feats, 'fps': fps, 'duration': duration, 'feat_stride': feat_stride,
                 'feat_num_frames': feat_num_frames}]
        features.append(data)
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    for iter_idx, video_list in enumerate(features, 0):
        # forward the model (wo. grad)
        #print(f'iter_idx {iter_idx}')
        with torch.no_grad():
            output = model(video_list)
            num_vids = len(output)


            for vid_idx in range(num_vids):
                results_per_video = []

                segments = output[vid_idx]['segments']
                scores = output[vid_idx]['scores']
                labels = output[vid_idx]['labels']
                for seg, label, score in zip(segments, labels, scores):
                    result = {
                        "label": Annotations[label.item()],
                        "segment": [seg[0].item(), seg[1].item()],  # Convert segment tensor to list
                        "score": score.item()
                    }

                    results_per_video.append(result)

                output_file = os.path.join(output_folder_directory, 'predictions',
                                           output[vid_idx]['video_id'] + '.json')
                directory = os.path.dirname(output_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results_per_video, f, ensure_ascii=False, indent=4)

                print(f"Results written to {output_file}")
if __name__ == '__main__':
    main()