import gc
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

def Network(conf='tf_efficientdet_d4', ckpt='efficientdet_d4-bb7e98fe.pth'):
    config = get_efficientdet_config(conf)
    model = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)
    config.num_classes = 1 
    config.image_size = 512
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001,
        momentum=.01))
    return DetBenchTrain(model, config)


def Inference(conf='tf_efficientdet_d4', ckpt='outputs/last-checkpoint.pth'):
    config = get_efficientdet_config(conf)
    model = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1 
    config.image_size = 512
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001,
        momentum=.01))

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = DetBenchPredict(model, config).eval()
    return model
