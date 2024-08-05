import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):

   def decorator(cls):
       #print(f'the name of the dataset : {name}')
       #print(f' cls : {cls}')
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, **kwargs):
    """
    A simple dataset builder
    """


    '''
    for i, data in enumerate(train_dataset):
        print("Batch Data Structure:")
        #print(type(data))
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

    dataset = datasets[name](is_training, split, **kwargs)




    print(datasets)

    return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
