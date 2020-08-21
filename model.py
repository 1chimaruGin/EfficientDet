import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def Network():
    config = get_efficientdet_config('tf_efficientdet_d4')
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('efficientdet_d4-5b370b7a.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)
