import os
import torch
import warnings
import logging
import time
from datetime import datetime

warnings.filterwarnings('ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class trainer:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 1

        self.base_dir = f'./{config.FOLDER}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_file = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model =model
        self.device = device

        _optimizer = list(self.model.named_parameters())
        non_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {'params': [param for name, param in _optimizer if not any(decay in name for decay in non_decay)], 'weight_decay': 0.001 },
            {'params': [param for name, param in _optimizer if any(decay in name for decay in non_decay)], 'weight_decay': 0.0 }
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = config.SCHEDULER_CLASS(self.optimizer, **config.SCHEDULER_PARAMS)
        self._log(f'Training Start on {self.device}')

    
    def fit(self, loader):
        for epoch in range(1, self.config.EPOCHS):
            if self.config.VERBOSE:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self._log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self._train_one_epoch(loader['train'])

            self._log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self._save(f'{self.base_dir}/last-checkpoint.pth')

            t = time.time()

            summary_loss = self._validation(loader['val'])

            self._log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self._save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.pth'))[:-3]:
                    os.remove(path)

            if self.config.VALIDATION_SCHEDULER:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def _validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.VERBOSE:
                if step % self.config.VERBOSE_STEP == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                img_scale = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
                img_size = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)

                output = self.model(images, {'bbox': boxes, 'cls': labels, 'image_scale': image_scale, 'img_size': img_size})
                loss = output['loss']
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss


    def _train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.VERBOSE:
                if step % self.config.VERBOSE_STEP == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()
            
            output = self.model(images, {'bbox': boxes, 'cls': labels})
            loss = output['loss']
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.STEP_SCHEDULER:
                self.scheduler.step()

        return summary_loss
    
    def _save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def _log(self, message):
        if self.config.VERBOSE:
            print(message)
        with open(self.log_file, 'a+') as logger:
            logger.write(f'{message}\n')

