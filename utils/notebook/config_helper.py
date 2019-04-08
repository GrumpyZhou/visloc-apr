import torch 
from utils.datasets.preprocess import get_transform_ops
from utils.common.setup import load_weights_to_gpu, make_deterministic
        
class AbsDummyConfig:
    def __init__(self, weights_dir, learn_weighting=True, beta=1.0, gpu=0):
        # Fixed params
        self.seed = 1
        self.batch_size = 75
        self.lr_init = 0.1
        self.optim = 'Adam'
        self.epsilon = 1.0
        self.weight_decay = 1e-3
        self.lr_decay = None
        self.optimizer_dict = None
        self.learn_weighting = learn_weighting 
        self.homo_init = [0.0, -3.0]
        self.beta = beta
        self.training = False
        self.ops = get_transform_ops(256, None, crop='center')
        
        # Customized params
        self.device = torch.device('cuda:{}'.format(gpu))
        self.weights_dir = weights_dir
        self.weights_dict = load_weights_to_gpu(weights_dir)