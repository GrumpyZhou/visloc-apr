import torch
import random
import numpy as np
from collections import OrderedDict

def load_weights_to_gpu(weights_dir=None, gpu=None):
    weights_dict = None
    if weights_dir is not None: 
        if gpu is not None:
            map_location = lambda storage, loc: storage.cuda(gpu)
        else:
            map_location = lambda storage, loc: storage
        weights = torch.load(weights_dir, map_location=map_location)
        if isinstance(weights, OrderedDict):
            weights_dict = weights
        elif isinstance(weights, dict) and 'state_dict' in weights:
            weights_dict = weights['state_dict']
    return weights_dict

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Important also
    
def lprint(ms, log=None):
    '''Print message to console and to a log file'''
    print(ms)
    if log:
        log.write(ms+'\n')
        log.flush()

def config_to_string(config, html=False):
    print_ignore = ['weights_dict', 'optimizer_dict']
    args = vars(config)
    separator = '<br>' if html else '\n' 
    confstr = ''
    confstr += '------------ Configuration -------------{}'.format(separator)
    for k, v in sorted(args.items()):
        if k in print_ignore:
            if v is not None:
                confstr += '{}:{}{}'.format(k, len(v), separator)
            continue
        confstr += '{}:{}{}'.format(k, str(v), separator)
    confstr += '----------------------------------------{}'.format(separator)
    return confstr

def cal_quat_angle_error(label, pred):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    d = np.abs(np.sum(np.multiply(q1,q2), axis=1, keepdims=True)) # Here we have abs()
    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error