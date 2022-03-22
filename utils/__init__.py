import json
import math
import numpy as np
import os
from pathlib import Path
import torch
from torch.optim import lr_scheduler

def transfer_to_device(x, device):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to_device(x[i], device)
    else:
        x = x.to(device)
    return x

def parse_configuration(config_file):
    if isinstance(config_file, str):
        with open(config_file) as json_file:
            return json.load(json_file)
    else:
        return config_file
    
def get_scheduler(optimizer, configuration, last_epoch=-1):
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=configuration['step_size'], gamma=0.3, last_epoch=last_epoch)
    else:
        return NotImplementedError('Learning rate policy {} is not implemented!'.format(configuration['lr_policy']))

def stack_all(list, dim=0):
    return [torch.stack(s, dim) for s in list]