import cv2
import torch
import numpy
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A 
from ensemble_boxes import weighted_boxes_fusion
from albumentations.pytorch.transforms import ToTensorV2

transform = A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def _make_predictions(model, images, score_threshold=0.22):
    predictions = []
    with torch.no_grad():
        det = model(images, torch.tensor([1]*images.shape[0]).float().cuda(), torch.tensor([images[0].shape[-2:]]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def _run_wbf(predictions, image_index, image_size=512, iou_thr=0.6, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def plot_bbox(model, image: str):
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(numpy.float32)
    image /= 255
    sample = {'image': image}
    image = transform(**sample)['image']
    image = image.cuda()
    predictions = _make_predictions(model, image.unsqueeze(0))

    i = 0
    sample = image.permute(1,2,0).cpu().numpy()

    boxes, scores, labels = _run_wbf(predictions, image_index=i)
    boxes = boxes.astype(np.int32).clip(min=0, max=511)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 1)
    
    ax.set_axis_off()
    ax.imshow(sample)
    plt.waitforbuttonpress()


