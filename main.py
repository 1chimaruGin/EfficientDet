import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SequentialSampler, RandomSampler
from data.preprocess import csv_to_dataset
from data.loader import create_custom_loader
from model import Network, Inference
from train import trainer
from config import TrainGlobalConfig
from utils import plot_bbox

parser = argparse.ArgumentParser(description='EfficientDet')

parser.add_argument('-m', '--mode', default='train', type=str, help='Mode')
parser.add_argument('-b', '--batch-size', default=4, type=int, help='Batch Size')
parser.add_argument('-nwk', '--num-workers', default=2, type=int, help='Num workers')
parser.add_argument('-p', '--path', default='data/train', type=str, help='Path to dataset')
parser.add_argument('-im', '--image', type=str, help='Image path')
parser.add_argument('-coef', '--coef', default=1, type=int, help='EfficientDet')
parser.add_argument('-w', '--load-weights', default='efficientdet_d4-5b370b7a.pth', type=str, help='load weights')
parser.add_argument('-r', '--resume', action='store_true', help='Resume training')
parser.add_argument('-out', '--output', type=str, default='outputs', help='Output directory')

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        model = Network(conf=f'tf_efficientdet_d{args.coef}', ckpt=args.load_weights)
        model.to(device)
        config = TrainGlobalConfig()
        config.FOLDER = args.output
        config.BATCH_SIZE = args.batch_size
        config.NUM_WORKERS = args.num_workers

        dataset = csv_to_dataset(path=args.path)

        loader = {x: create_custom_loader(dataset[x], batch_size=config.BATCH_SIZE, 
                num_workers=config.NUM_WORKERS) for x in ['train', 'val']}

        trainer = trainer(model, device=device, config=config, resume=args.resume)
        trainer.fit(loader)

    elif args.mode == 'predict':
        model = Inference(conf=f'tf_efficientdet_d{args.coef}', ckpt='outputs/best-checkpoint-039epoch.bin')
        model.to(device)
        plot_bbox(model.eval(), image='data/test/2fd875eaa.jpg')       


        


