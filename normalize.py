import torch
from data.preprocess import csv_to_dataset
from data.loader import create_custom_loader

dataset = csv_to_dataset(path='data/train.csv')
loader = {x: create_custom_loader(dataset[x], batch_size=32, 
                num_workers=0) for x in ['train', 'val']}

def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X**2]
    channel_sum, channel_square_sum, num_batches = 0, 0, 0

    for data, _, _ in loader['val']:
        channel_sum += torch.mean(data.permute(1,2,0), dim=[0, 2, 3])
        channel_square_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channel_sum / num_batches
    std = (channel_square_sum / num_batches - mean**2)**0.5

    return mean, std

mean, std = get_mean_std(loader)
print(mean, std)