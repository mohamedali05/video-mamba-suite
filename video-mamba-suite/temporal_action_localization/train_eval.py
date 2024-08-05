# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import json

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


def adjust_state_dict(state_dict, num_classes):
    conv_weight = state_dict['module.cls_head.cls_head.conv.weight']
    conv_bias = state_dict['module.cls_head.cls_head.conv.bias']

    if conv_weight.shape[0] != num_classes:
        state_dict['module.cls_head.cls_head.conv.weight'] = torch.nn.functional.pad(
            conv_weight[:num_classes], (0, 0, 0, 0, 0, max(0, num_classes - conv_weight.shape[0]))
        )
        state_dict['module.cls_head.cls_head.conv.bias'] = torch.nn.functional.pad(
            conv_bias[:num_classes], (0, max(0, num_classes - conv_bias.shape[0]))
        )
    return state_dict


################################################################################

def save_stats(ap, mAP , Total_mean_AP, f1 , mF1, total_mean_f1 , ckpt_folder, num_classes) :
    if num_classes == 3 :
        Annotations = ["OTHER", "SERVICE", "ECHANGE"]
    else :
        Annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]

    print(Annotations)
    IoU = [0.3, 0.4, 0.5, 0.6, 0.7]

    dictionnaries_result = {f'IoU {iou} : ': {} for iou in IoU}
    for i, IoU in enumerate(dictionnaries_result):
        annotations_dict = {annotation: {'AP: ': ap[i][j], 'F1: ': f1[i][j]} for j, annotation in
                            enumerate(Annotations)}
        dictionnaries_result[IoU] = annotations_dict
        dictionnaries_result[IoU]['MAP'] = mAP[i]
        dictionnaries_result[IoU]['F1'] = mF1[i]

    dictionnaries_result['total mean Avearage Precision '] = Total_mean_AP
    dictionnaries_result['F1 score Mean '] = total_mean_f1
    #print(dictionnaries_result)

    results_file = os.path.join(ckpt_folder, 'ap_per_class_new_model.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(dictionnaries_result, f, ensure_ascii=False, indent=4)


def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""

    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    #pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if len(args.output) == 0:
        ckpt_folder = os.path.join(cfg['output_folder'], f"{cfg_filename}_{ts}")
    else:
        # this is what we are actually doing with the tennis database
        ckpt_folder = os.path.join(
            cfg['output_folder'],
            f"{ts}_{args.output}")
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer

    results_file_path = os.path.join(ckpt_folder, "Results.txt")
    best_results_file_path = os.path.join(ckpt_folder, "best.txt")
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )

    #print(f"the training dataset {train_dataset}")


    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()

    #print(f"train_db_vars  :type {type(train_db_vars)} , data : {train_db_vars} ; ")
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])





    """ Print for understanding the dataset :"""

    '''
    print (f'length of the train dataset  {len(train_dataset)}')

    for i, data in enumerate(train_dataset):

        print("Batch Data Structure:")
        print(type(data))
        print(f"Video IDs: {data['video_id']}")
        print(f"Feature Tensor Shape: {data['feats'].shape}")
        print(f"Segments Shape: {data['segments'].shape}")
        print(f"Labels Shape: {data['labels'].shape}")
        print(f"FPS: {data['fps']}")
        print(f"Duration: {data['duration']}")
        print(f"Feature Stride: {data['feat_stride']}")
        print(f"Number of Feature Frames: {data['feat_num_frames']}")
        print(data)
        if (i > 5):
            break
            
    '''







    """ Print for understanding the dataset :"""

    '''
    for i,data in enumerate(train_loader):

        print("Batch Data Structure:")
        print (f" Type : {type(data)}")
        print(f"length of the list {len(data)}")
        print(data)
        
        print(f"Video IDs: {data[0]['video_id']}")
        print(f"Feature Tensor Shape: {data[0]['feats'].shape}")
        print(f"Segments Shape: {data[0]['segments'].shape}")
        print(f"Labels Shape: {data[0]['labels'].shape}")
        print(f"FPS: {data[0]['fps']}")
        print(f"Duration: {data[0]['duration']}")
        print(f"Feature Stride: {data[0]['feat_stride']}")
        print(f"Number of Feature Frames: {data[0]['feat_num_frames']}")
        print(data[0])
        
        if (i > 5) : 
            break
        
    '''






    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    #print(model)
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))

            print(f'checkpoint : {type(checkpoint)}')
            #print(f'checkpoint keys {checkpoint.keys()}')

            '''
            #modifying the checkpoint to have the correct values of data
            checkpoint['state_dict_ema']['module.cls_head.cls_head.conv.weight'] = checkpoint['state_dict_ema']['module.cls_head.cls_head.conv.weight'][:11]
            checkpoint['state_dict_ema']['module.cls_head.cls_head.conv.bias'] = checkpoint['state_dict_ema']['module.cls_head.cls_head.conv.bias'][:11]
            checkpoint['state_dict']['module.cls_head.cls_head.conv.weight'] = checkpoint['state_dict']['module.cls_head.cls_head.conv.weight'][:11]
            checkpoint['state_dict']['module.cls_head.cls_head.conv.bias'] = checkpoint['state_dict']['module.cls_head.cls_head.conv.bias'][:11]
            '''
            current_num_classes = cfg['dataset']['num_classes']
            # Adjust state_dicts
            checkpoint['state_dict_ema'] = adjust_state_dict(checkpoint['state_dict_ema'], current_num_classes)
            checkpoint['state_dict'] = adjust_state_dict(checkpoint['state_dict'], current_num_classes)
            #print(list(checkpoint['state_dict_ema'].keys()))

            print(checkpoint['state_dict_ema']['module.cls_head.cls_head.conv.weight'].shape)

            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            for param in model.module.parameters():
                param.requires_grad = False

            for param in model.module.cls_head.parameters():
                param.requires_grad = True


            optimizer = make_optimizer(model, cfg['opt'])
            scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )


    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )



    best_mAP = 0.0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
        )


        if epoch>=1:#(max_epochs//4):


        # if epoch>1:#(max_epochs//3):

            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg['devices'])


            model_eval.load_state_dict(model_ema.module.state_dict())


            # set up evaluator
            det_eval, output_file = None, None
            # if not args.saveonly:
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds = val_db_vars['tiou_thresholds']
            )
            # else:
            #     output_file = os.path.join('eval_results.pkl')

            """5. Test the model"""
            print("\nStart testing model {:s} ...".format(cfg['model_name']))


            ########################

            ap, mAP, Total_mean_AP, f1, mF1, total_mean_f1 = valid_one_epoch(
                val_loader,
                model_eval,
                curr_epoch= epoch,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=999999 ,  #args.print_freq
                file_path = results_file_path ,
                best_file_path = best_results_file_path ,
                best_mAP = best_mAP

            )
            if epoch == 1 :
                best_mAP = Total_mean_AP



            ########################
            if Total_mean_AP > best_mAP :

                best_mAP = Total_mean_AP
                save_ckpt = True
                save_stats(ap, mAP, Total_mean_AP, f1, mF1, total_mean_f1, ckpt_folder, cfg['dataset']['num_classes'])
            else :
                save_ckpt = False

            end = time.time()
            # print("All done! Total time: {:0.2f} sec".format(end - start))
            #print(epoch,mAP)

            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                # was initially true
                save_ckpt,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}_{:.5f}.pth.tar'.format(epoch,Total_mean_AP)
            )




    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser

    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--name_ckpt', default='', type=str,
                        help='name the ckpt (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
