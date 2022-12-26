# encoding: utf-8
from torch.optim import lr_scheduler


def make_scheduler(scheduler_type, optimizer):
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'one_cycle_cos':
        return lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
    elif scheduler_type == 'cosine_annealing':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

    raise "LR scheduler type not supported"
