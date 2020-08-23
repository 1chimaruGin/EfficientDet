import torch

class TrainGlobalConfig:

    NUM_WORKERS = 2

    BATCH_SIZE = 4 

    EPOCHS = 40 

    LEARNING_RATE = 0.0002

    FOLDER = 'outputs'

    VERBOSE = True

    VERBOSE_STEP = 1

    STEP_SCHEDULER = False  # do scheduler.step after optimizer.step
    
    VALIDATION_SCHEDULER = True  # do scheduler.step after validation stage loss
    
    SCHEDULER_CLASS = torch.optim.lr_scheduler.ReduceLROnPlateau

    SCHEDULER_PARAMS = dict(
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

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")