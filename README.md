# EfficientDet

The [EfficientDet](https://github.com/rwightman/efficientdet-pytorch) and [Backbone](https://github.com/rwightman/pytorch-image-models) are from the amazing repo of [rwightman](https://github.com/rwightman). Check his repo to learn more about other amazing architectures.

Special thanks to [Alex Shonenkov](https://www.kaggle.com/shonenkov) for providing great [kernel](https://www.kaggle.com/shonenkov/training-efficientdet). Don't forget to upvote his work :3.

## Training

Training EfficientDet is a painful and time-consuming task.

- step 1 -

```
# your dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, coco2017
datasets/
    -coco2017/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```

- step 2 -

`python main.py -m train`

Coming Soon!