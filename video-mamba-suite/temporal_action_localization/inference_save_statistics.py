"""
This script is designed to evaluate a pre-trained video action localization model on a dataset of tennis games.
 It performs the following key tasks:

1. **Configuration and Model Loading:**
   - Loads model configuration and weights from specified paths.
   - Sets up the model for evaluation using multiple GPUs if available.

2. **Dataset and DataLoader Setup:**
   - Configures the dataset and data loader for the evaluation, specifying the JSON file with ground truth annotations and the feature folder.

3. **Model Inference:**
   - Runs the model in evaluation mode over the dataset to generate predictions, including start and end times of actions, their labels, and scores.

4. **Evaluation and Metrics Calculation:**
   - Uses the `ANETdetection` class to evaluate the predictions against the ground truth annotations.
   - Calculates metrics such as Average Precision (AP), F1 scores at different Intersection over Union (IoU) thresholds, and aggregates these into a mean Average Precision (mAP) and a mean F1 score.

5. **Results Saving:**
   - Outputs the evaluation results, including detailed per-class metrics and overall performance, to a JSON file.
"""


# python import
import os
from inference_pipeline_video_tennis import load_config_from_txt
# torch imports
import torch
import torch.nn as nn
import torch.utils.data
import json
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection

def main() :

    model_path = './ckpt_tennis/2024-07-08_16-51-19_best_params_3_classes'  # path to the model
    config_path = os.path.join(model_path, 'config.txt')
    ckpt_path = os.path.join(model_path, 'model_best.pth.tar')
    cfg = load_config_from_txt(config_path)


    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])



    checkpoint = torch.load(ckpt_path,
                    map_location = lambda storage, loc: storage.cuda(
                        cfg['devices'][0]))


    model.load_state_dict(checkpoint['state_dict'] )
    # Configure out here the dataset that we want to test out and the feature folder
    cfg['dataset']['json_file'] =  './data/Tennis/annotations/All_test_games_ground_truth.json'
    cfg['dataset']['feat_folder'] =  os.path.join('our_dataset','test_set', 'game_features')
    output_folder_directory = os.path.join('results', 'Results_test_set')

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
                    val_dataset.json_file,
                    val_dataset.split[0],
                    tiou_thresholds = val_db_vars['tiou_thresholds']
                )

    if cfg['dataset']['num_classes'] == 11:
        Annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
    else:
        Annotations = ['OTHER', 'SERVICE', 'EXCHANGE']

    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)

    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }



    #configure out the results file for the ap_per_class.json
    results_file = os.path.join(output_folder_directory,'ap_per_class.json')
    directory = os.path.dirname(results_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)

        with torch.no_grad():
            output = model(video_list)

            num_vids = len(output)
            for vid_idx in range(num_vids) :
                results['video-id'].extend(
                    [output[vid_idx]['video_id']] *
                    output[vid_idx]['segments'].shape[0]
                )
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    ap ,f1 = det_eval.evaluate(results, verbose=False ,inference=True)

    mAP = ap.mean(axis=1)
    Tot_mean_average = mAP.mean()
    mF1= f1.mean(axis=1)
    Tot_mean_F1 = mF1.mean()
    IoU = [0.3 , 0.4 , 0.5 , 0.6 , 0.7]
    dictionnaries_result = {f'IoU {iou} : ' : {} for iou in IoU }
    for i, IoU in enumerate(dictionnaries_result) :
        annotations_dict = {annotation:{ 'AP: ' : ap[i][j] , 'F1: ' : f1[i][j] } for j, annotation in enumerate(Annotations)}
        dictionnaries_result[IoU] = annotations_dict
        dictionnaries_result[IoU]['MAP'] =  mAP[i]
        dictionnaries_result[IoU]['F1'] = mF1[i]

    dictionnaries_result['total mean Avearage Precision '] = Tot_mean_average
    dictionnaries_result['F1 score Mean '] = Tot_mean_F1
    #print(dictionnaries_result)
    #print(directory)

    with open(results_file, 'w' , encoding='utf-8') as f:
        json.dump(dictionnaries_result, f,ensure_ascii=False ,  indent=4)
    print(f'Results written to {results_file}')



if __name__ == '__main__':
    main()


