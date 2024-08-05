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
from torchsummary import summary
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)



import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming load_config and make_meta_arch functions are defined elsewhere in your codebase



'''
# Load the config
cfg = load_config('./configs/mamba_thumos_new.yaml')

# Create the model
model = make_meta_arch(cfg['model_name'], **cfg['model'])
model = nn.DataParallel(model, device_ids=cfg['devices'])

# Load the checkpoint
checkpoint_path = './ckpt_thumos/mamba_thumos_new_mamba_thumos_2_0.0001/model_best.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda('cuda:0'))

# Load the state dictionary into the model
model.load_state_dict(checkpoint['state_dict'])

# Print the model architecture
print(model)
print('####################################')
# Extract the module from DataParallel
model = model.module

print(model)



# If necessary, provide further information about the preprocessing function


'''


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

    num_new_classes = 11  # Change this to the desired number of classes



    # Print the modified model architecture
    print("\nModified model architecture:\n")
    print(model)

    # Load the checkpoint
    checkpoint = torch.load('./ckpt_thumos/mamba_thumos_new_mamba_thumos_2_0.0001/model_best.pth.tar',
                            map_location=lambda storage, loc: storage.cuda('cuda:0'))
    model.load_state_dict(checkpoint['state_dict'])

    model = model.module
    # Access the classification head
    cls_head = model.cls_head.cls_head

    new_cls_head = nn.Conv1d(in_channels=cls_head.conv.in_channels,
                             out_channels=num_new_classes,
                             kernel_size=cls_head.conv.kernel_size,
                             stride=cls_head.conv.stride,
                             padding=cls_head.conv.padding,
                             bias=cls_head.conv.bias is not None)

    # Replace the old classification head with the new one
    model.cls_head.cls_head.conv = new_cls_head

    # Optionally, freeze all layers except the new classification head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.cls_head.parameters():
        param.requires_grad = True

    print(model)

    # Print the model summary using a custom function
    #print_model_summary(model.module)

if __name__ == '__main__':
    main()
