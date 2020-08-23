import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from data.preprocess import csv_to_dataset
from data.loader import create_custom_loader
from model import Network
from train import trainer
from config import TrainGlobalConfig


parser = argparse.ArgumentParser(description='EfficientDet')

parser.add_argument('-m', '--mode', default='train', type='str', help='Mode')

args = parser.parse_args()

if __name__ == '__main__':
    from model import Network

    model = Network()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    config = TrainGlobalConfig()

    dataset = csv_to_dataset(path='data/train.csv')

    loader = {x: create_custom_loader(dataset[x], batch_size=config.BATCH_SIZE, 
                num_workers=config.NUM_WORKERS) for x in ['train', 'val']}

    trainer = trainer(model, device=device, config=config)
    trainer.fit(loader)

        


