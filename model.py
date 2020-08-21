import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def Network(conf='efficientdet_d1'):
    config = get_efficientdet_config(conf)
    model = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('efficientdet_d1-bb7e98fe.pth')
    model.load_state_dict(checkpoint)
    config.num_classes = 1 
    config.image_size = 512
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001,
        momentum=.01))
    return DetBenchTrain(model, config)
