import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from data.preprocess import csv_to_dataset
from data.loader import create_custom_loader
from model import Network
from train import trainer
dataset = csv_to_dataset(path='data/train.csv')

loader = {x: create_custom_loader(dataset[x], batch_size=4) for x in ['train', 'val']}

'''
img, trg, _ = next(iter(loader['val']))

image = img[0].permute(1,2,0).cpu().numpy()

boxes = trg[0]['boxes'].cpu().numpy().astype(np.int32)

fig, ax = plt.subplots(1, 1, figsize=(16, 10))

for box in boxes:
    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2)

ax.set_axis_off()
ax.imshow(image)
plt.waitforbuttonpress()
'''


class TrainGlobalConfig:
    num_workers = 0
    batch_size = 4 
    n_epochs = 3 # n_epochs = 40
    lr = 0.0002

    folder = 'outputs'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

#     SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
#     scheduler_params = dict(
#         max_lr=0.001,
#         epochs=n_epochs,
#         steps_per_epoch=int(len(train_dataset) / batch_size),
#         pct_start=0.1,
#         anneal_strategy='cos', 
#         final_div_factor=10**5
#     )
    
    schedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )

if __name__ == '__main__':
    from model import Network

    model = Network()
    device = torch.device('cuda')
    model.to(device)
    config = TrainGlobalConfig()
    trainer = trainer(model, device=device, config=config)
    trainer.fit(loader)

        


