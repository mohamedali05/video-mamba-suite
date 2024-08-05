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
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

################################################################################
def modify_classification_head(model, num_new_classes):
    # Modify the classification head for the new number of classes
    old_cls_head = model.cls_head.cls_head
    new_cls_head = nn.Conv1d(in_channels=old_cls_head.conv.in_channels,
                             out_channels=num_new_classes,
                             kernel_size=old_cls_head.conv.kernel_size,
                             stride=old_cls_head.conv.stride,
                             padding=old_cls_head.conv.padding,
                             bias=old_cls_head.conv.bias is not None)
    model.cls_head.cls_head = new_cls_head

    # Freeze all layers except the classification head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.cls_head.parameters():
        param.requires_grad = True

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
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
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # Modify the classification head for the new number of classes
    num_new_classes = 11  # Change this to the desired number of classes
    modify_classification_head(model, num_new_classes)

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Load Pre-trained Model"""
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            # load pre-trained weights
            checkpoint = torch.load(args.pretrained,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))

            # Load state dict while ignoring the missing classification head weights
            model_state_dict = model.state_dict()
            pretrained_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
            model_state_dict.update(pretrained_state_dict)
            model.load_state_dict(model_state_dict)

            # Load EMA state dict if available
            if 'state_dict_ema' in checkpoint:
                model_ema_state_dict = model_ema.module.state_dict()
                pretrained_ema_state_dict = {k: v for k, v in checkpoint['state_dict_ema'].items() if k in model_ema_state_dict}
                model_ema_state_dict.update(pretrained_ema_state_dict)
                model_ema.module.load_state_dict(model_ema_state_dict)

            print("=> loaded pre-trained model from '{}'".format(args.pretrained))
            del checkpoint
        else:
            print("=> no pre-trained model found at '{}'".format(args.pretrained))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))
    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    save_ckpt = False
    best_mAP = 0.0

    # Training loop
    for epoch in range(args.start_epoch, max_epochs):
        # Train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        if epoch >= 1:
            # model
            model_eval = make_meta_arch(cfg['model_name'], **cfg['model'])
            # not ideal for multi GPU training, ok for now
            model_eval = nn.DataParallel(model_eval, device_ids=cfg['devices'])
            model_eval.load_state_dict(model_ema.module.state_dict())

            # set up evaluator
            det_eval, output_file = None, None
            val_db_vars = val_dataset.get_attributes()
            det_eval = ANETdetection(
                val_dataset.json_file,
                val_dataset.split[0],
                tiou_thresholds=val_db_vars['tiou_thresholds']
            )

            """6. Test the model"""
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()

            mAP = valid_one_epoch(
                val_loader,
                model_eval,
                curr_epoch=epoch,
                evaluator=det_eval,
                output_file=output_file,
                ext_score_file=cfg['test_cfg']['ext_score_file'],
                tb_writer=None,
                print_freq=999999,
                file_path=results_file_path,
                best_file_path=best_results_file_path,
                best_mAP=best_mAP
            )
            if epoch == 1:
                best_mAP = mAP

            if mAP > best_mAP:
                best_mAP = mAP
                save_ckpt = True
            else:
                save_ckpt = False

            end = time.time()
            print(epoch, mAP)

            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                save_ckpt,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}_{:.5f}.pth.tar'.format(epoch, mAP)
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
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to a pre-trained model (default: none)')
    args = parser.parse_args()
    main(args)
