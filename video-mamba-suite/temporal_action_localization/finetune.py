


"""
This script fine-tunes a pre-trained video action localization model
on the Thumos dataset for a new dataset (the tennis dataset) with a different number of classes.
"""

# python imports
import torch.utils.data
from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)



import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming load_config and make_meta_arch functions are defined elsewhere in your codebase


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


def print_model_summary(model):
    def recursive_summary(model, indent=0):
        for name, module in model.named_children():
            print('  ' * indent + f'{name}: {module.__class__.__name__}')
            recursive_summary(module, indent + 1)

    print("Model architecture:\n")
    print(model)
    print("\nDetailed model summary:\n")
    recursive_summary(model)

def main() :
    # Load the config
    cfg = load_config('./configs/mamba_thumos_new.yaml')

    # Create the model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    model_ema = ModelEma(model)
    num_new_classes = 11  # Change this to the desired number of classes

    num_iters_per_epoch = cfg['loader']['batch_size']

    # Print the modified model architecture
    print("\nModified model architecture:\n")
    print(model)

    # Load the checkpoint
    checkpoint = torch.load('./ckpt_thumos/mamba_thumos_new_mamba_thumos_2_0.0001/model_best.pth.tar',
                            map_location=lambda storage, loc: storage.cuda('cuda:0'))
    model.load_state_dict(checkpoint['state_dict'])
    current_num_classes = 11
    # Adjust state_dicts
    checkpoint['state_dict_ema'] = adjust_state_dict(checkpoint['state_dict_ema'], current_num_classes)
    checkpoint['state_dict'] = adjust_state_dict(checkpoint['state_dict'], current_num_classes)

    model_ema.module.load_state_dict(checkpoint['state_dict_ema'])

    # Optionally, freeze all layers except the new classification head
    for param in model.module.parameters():
        param.requires_grad = False

    for param in model.module.cls_head.parameters():
        param.requires_grad = True
    optimizer = make_optimizer(model, cfg['opt'])
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    save_states = {
        'epoch': 0,
        'state_dict': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    save_states['state_dict_ema'] = model_ema.module.state_dict()
    save_checkpoint(save_states , True , file_folder= 'finetuned_model', file_name='model_finetuned.tar')



    # Print the model summary using a custom function
    #print_model_summary(model.module)

if __name__ == '__main__':
    main()
