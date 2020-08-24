import os
import torch
import pandas as pd 
import numpy as np
from glob import glob
from sklearn.model_selection import StratifiedKFold
from data.dataset import Custom_Dataset
from data.transforms import get_train_transforms, get_valid_transforms


def csv_to_dataset(path=None, split=5):
    data = pd.read_csv(path+'.csv')
    bbox = np.stack(data.bbox.apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        data[column] = bbox[:, i]
    data.drop(columns=['bbox'], inplace=True)

    skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=42)

    fold_data = data[['image_id']].copy()
    fold_data.loc[:, 'bbox_count'] = 1
    fold_data = fold_data.groupby('image_id').count()
    fold_data.loc[:, 'source'] = data[['image_id', 'source']].groupby('image_id').min()['source']
    fold_data.loc[:, 'stratify_group'] = np.char.add(fold_data.source.values.astype(str), 
                                                  fold_data.bbox_count.apply(lambda x: f'_{x // 15}').values.astype(str))

    fold_data.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=fold_data.index, y=fold_data['stratify_group'])):
        fold_data.loc[fold_data.iloc[val_index].index, 'fold'] = fold_number


    fold_number = 0

    train_dataset = Custom_Dataset(
        root= path,
        image_ids=fold_data[fold_data['fold'] != fold_number].index.values,
        data=data,
        transform=get_train_transforms(),
        test=False,
    )

    validation_dataset = Custom_Dataset(
        root= path,
        image_ids=fold_data[fold_data['fold'] == fold_number].index.values,
        data=data,
        transform=get_valid_transforms(),
        test=True,
    )

    return {
        'train': train_dataset,
        'val': validation_dataset
    }