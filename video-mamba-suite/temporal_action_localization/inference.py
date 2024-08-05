# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

import json

# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


def main(args) :

    config_path = './configs/mamba_tennis_new.yaml'
    cfg = load_config(config_path)
    model_path = './ckpt_tennis/2024-07-08_16-51-19_best_params_3_classes'  # path to the model
    config_path = os.path.join(model_path, 'config.txt')
    ckpt_path = os.path.join(model_path, 'model_best.pth.tar')
    cfg = load_config_from_txt(config_path)



    # model
    '''
    Annotations = ["PHASE DE NON-JEU",
                   "Service réussi du joueur éloigné de la caméra",
                   "Service fautif du joueur éloigné de la caméra",
         "Service avec let (rencontre du filet) du joueur éloigné de la caméra",
                   "Service réussi du joueur proche de la caméra",
                   "Service fautif du joueur proche de la caméra",
                   "Service avec let (rencontre du filet) du joueur proche de la caméra",
                   "COUP DU JOUEUR FACE A LA CAMERA : RECEPTION DE SON COTE DROIT",
                   "COUP DU JOUEUR FACE A LA CAMERA : RECEPTION DE SON COTE GAUCHE",
                   "COUP DU JOUEUR DE DOS: RECEPTION COTE GAUCHE",
                   "COUP DU JOUEUR DE DOS: RECEPTION COTE DROIT"]
                   
    '''
                   
    

    #Annotations = ['OTHER', 'SERVICE', 'EXCHANGE']


    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])



    checkpoint = torch.load('./ckpt_tennis/2024-07-08_16-51-19_best_params_3_classes/model_best.pth.tar',
                    map_location = lambda storage, loc: storage.cuda(
                        cfg['devices'][0]))


    '''
    checkpoint = torch.load('./ckpt_tennis/2024-07-25_11-41-04_tennis_hayden_set/model_best.pth.tar',
                            map_location=lambda storage, loc: storage.cuda(
                                cfg['devices'][0]))


    '''

    model.load_state_dict(checkpoint['state_dict'] )


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

    if args.save_statistics :
        print('save_statistics : ')


        for iter_idx, video_list in enumerate(val_loader, 0):
            # forward the model (wo. grad)
            print(f'iteration number = {iter_idx}')
            with torch.no_grad():
                output = model(video_list)
                print('model output: ', output)
                num_vids = len(output)
                for vid_idx in range(num_vids) :
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    print("output shape : ")
                    print(output[vid_idx]['segments'].shape[0])
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])

                str_res_shape = len(results['t-start'])
                print(f'results shape {str_res_shape}')

        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()




        ap ,f1 = det_eval.evaluate(results, verbose=False ,inference=True)

        mAP = ap.mean(axis=1)
        Tot_mean_average = mAP.mean()
        mF1= f1.mean(axis=1)
        Tot_mean_F1 = mF1.mean()

        #print(f'confusion matrix shape : {cm.shape} \n confusion matrix : \n{cm} ')


        #print(f' average precision per class  : {ap}')

        if cfg['dataset']['num_classes'] == 11 :
            Annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
        else :
            Annotations = ['OTHER','SERVICE','EXCHANGE']
        IoU = [0.3 , 0.4 , 0.5 , 0.6 , 0.7]




        dictionnaries_result = {f'IoU {iou} : ' : {} for iou in IoU }
        for i, IoU in enumerate(dictionnaries_result) :
            annotations_dict = {annotation:{ 'AP: ' : ap[i][j] , 'F1: ' : f1[i][j] } for j, annotation in enumerate(Annotations)}
            dictionnaries_result[IoU] = annotations_dict
            dictionnaries_result[IoU]['MAP'] =  mAP[i]
            dictionnaries_result[IoU]['F1'] = mF1[i]

        dictionnaries_result['total mean Avearage Precision '] = Tot_mean_average
        dictionnaries_result['F1 score Mean '] = Tot_mean_F1




        results_file = args.output_file_path
        directory = os.path.dirname(results_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print(directory)

        with open(results_file, 'w' , encoding='utf-8') as f:
            json.dump(dictionnaries_result, f,ensure_ascii=False ,  indent=4)



    if args.save_results :
        print('save_results : ')
        for iter_idx, video_list in enumerate(val_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():
                output = model(video_list)
                num_vids = len(output)
                for vid_idx in range(num_vids) :
                    results_per_video = []

                    video_id = output[vid_idx]['video_id']
                    segments = output[vid_idx]['segments']
                    scores = output[vid_idx]['scores']
                    labels = output[vid_idx]['labels']
                    for seg, label, score in zip(segments, labels, scores):
                        result = {
                            "label": 'SERVICE' if 1 <= label.item() <= 6 else Annotations[label.item()],
                            "segment": [seg[0].item(), seg[1].item()],  # Convert segment tensor to list
                            "score": score.item()
                        }

                        results_per_video.append(result)
                        print(f' length of the results : {len(results_per_video)}')
                    output_file = os.path.join('results', output[vid_idx]['video_id'] + '.json')

                    with open(output_file, 'w' , encoding='utf-8') as f:
                        json.dump(results_per_video, f,ensure_ascii=False ,  indent=4)


                    print(f"Results written to {output_file}")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_out =  os.path.join('results','Results_test_set',  'ap_per_class.json')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_statistics', action='store_true')
    parser.add_argument('--output_file_path', default = default_out )
    args = parser.parse_args()
    main(args)


