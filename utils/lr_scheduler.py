import torch.optim as optim


def initialize_lr_scheduler(scheduler_name, optimizer):
    if scheduler_name == 'ReduceLROnPlateau':
           lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    elif scheduler_name == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_name == 'MultiStepLR':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
    elif scheduler_name == 'ExponentialLR':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        raise ValueError(f"Unknown learning rate scheduler: {scheduler_name}. \n",
                             f"Please choose 'ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR', or 'CosineAnnealingLR'.")
    return lr_scheduler