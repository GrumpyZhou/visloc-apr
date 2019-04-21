import torch
import random
from torchvision import transforms
import numpy as np
from PIL import Image, ImageChops

def get_transform_ops(resize=256, image_mean=None, crop='center', crop_size=224, normalize=False):
    ops = []
    if resize:
        ops.append(transforms.Resize(resize, Image.BICUBIC))
    if image_mean is not None:
        ops.append(MeanSubtractNumpy(np.load(image_mean)))
    if crop == 'center':
        crop = CenterCropNumpy(crop_size)
        ops.append(crop)       
    elif crop == 'random':
        crop = RandomCropNumpy(crop_size)
        ops.append(crop)
    if normalize:
        ops.append(ToTensorScaled())  # Scale value to [0, 1] 
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        ops.append(ToTensorUnscaled())
    return transforms.Compose(ops)

class ToTensorScaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        im /= 255.0 
        return torch.from_numpy(im)

    def __repr__(self):
        return 'ToTensorScaled(./255)'

class ToTensorUnscaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor'''
    def __call__(self, im):    
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return 'ToTensorUnscaled()'
        
class MeanSubtractPIL(object):
    '''Mean subtract operates on PIL Images'''
    def __init__(self, im_mean):
        self.im_mean = im_mean
       
    def __call__(self, im):
        if self.im_mean is None:
            return im
        return ImageChops.subtract(im, self.im_mean) 

    def __repr__(self):
        if self.im_mean is None:
            return 'MeanSubtractNumpy(None)'
        return 'MeanSubtractNumpy(im_mean={})'.format(self.im_name.filename)

class MeanSubtractNumpy(object):
    '''Mean subtract operates on numpy ndarrays'''
    def __init__(self, im_mean):
        self.im_mean = im_mean
        
    def __call__(self, im):
        if self.im_mean is None:
            return im
        return np.array(im).astype('float') - self.im_mean.astype('float')

    def __repr__(self):
        if self.im_mean is None:
            return 'MeanSubtractNumpy(None)'
        return 'MeanSubtractNumpy(im_mean={})'.format(self.im_mean.shape)

class CenterCropNumpy(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, im):
        im = np.array(im)
        size = self.size    
        h, w, _ = im.shape
        if w == size and h == size:
            return im
        x = int(round((w - size) / 2.))
        y = int(round((h - size) / 2.))
        return im[y:y+size, x:x+size, :]
    
    def __repr__(self):
        return 'CenterCropNumpy(size={})'.format(self.size)    
    
class RandomCropNumpy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = np.array(im)
        size = self.size
        h, w, _ = im.shape
        if w == size and h == size:
            return im
        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)
        return im[y:y+size, x:x+size, :]

    def __repr__(self):
        return 'RandomCropNumpy(size={})'.format(self.size) 
