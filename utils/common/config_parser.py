import argparse
import os
import torch

class AbsPoseConfig:
    def __init__(self):
        description = 'Absolute Pose Regression'   
        parser = argparse.ArgumentParser(description=description)
        self.parser = parser
        
        # Add different groups for arguments
        prog_group = parser.add_argument_group('General Program Config') 
        data_group = parser.add_argument_group('Data Loading Config', 'Options regarding image loading and preprocessing')
        model_group = parser.add_argument_group('Model Config', 'Options regarding network model and optimization')
        visdom_group = parser.add_argument_group('Visdom Config', 'Options regarding visdom server for visualization')

        # Program general settings 
        prog_group.add_argument('--test', action='store_false', dest='training', help='set program to a testing phase')
        prog_group.add_argument('--train', action='store_true', dest='training', help='set program to a training phase')     
        prog_group.add_argument('--validate', '-val', metavar='%d[N]', type=int, nargs=1, 
                                     help='evaluate model every N epochs during training')  
        prog_group.add_argument('--pretrained', metavar='%s',  type=str, nargs=1, default=None,
                                     help='the pre-trained weights to initialize the model(default: %(default)s)')
        prog_group.add_argument('--resume', metavar='%s',  type=str, nargs=1, default=None, 
                                     help='the checkpoint file to reload(default: %(default)s)')
        prog_group.add_argument('--seed', '-s', metavar='%d', type=int, default=1, 
                                     help='seed for randomization(default: %(default)s)')
        prog_group.add_argument('--odir', '-o', metavar='%s', type=str, required=True, 
                                     help='directory for program outputs')
        prog_group.add_argument('--gpu', metavar='%d',  type=int, default=0, 
                                     help='gpu device to use(cpu used if no available gpu)(default: %(default)s)')
      
        # Data loading and preprocess
        data_group.add_argument('--data_root', '-root', metavar='%s', type=str, default='data',
                                         help='the root directory containing target datasets(default: %(default)s)' )
        data_group.add_argument('--dataset', '-ds', metavar='%s', type=str, required=True, 
                                         help='the target dataset under data root' )        
        data_group.add_argument('--pose_txt',  metavar='%s', default='dataset_train.txt', 
                                         help='the file to load pose labels(default: %(default)s)')
        data_group.add_argument('--val_pose_txt', metavar='%s', type=str, default='dataset_test.txt', 
                                         help='the file to load validation image pose labels(default: %(default)s)')     
        
        data_group.add_argument('--batch_size', '-b', metavar='%d', type=int, default=75, 
                                    help='batch size to load the image data(default: %(default)s)')
        data_group.add_argument('--num_workers', '-n', metavar='%d', type=int, default=0, 
                                    help='number of subprocesses for data loading(default: %(default)s)')
        data_group.add_argument('--image_mean', '-imean',  metavar='%s',  type=str, default=None, 
                                    help='path of image_mean file name relative to the dataset path(default: %(default)s)')
        data_group.add_argument('--rescale', '-rs', metavar='%d', type=int, default=256, 
                                    help='target size (the shorter edge) after rescaling(default: %(default)s)')
        data_group.add_argument('--crop', '-c',  metavar='%d', type=int, default=224, 
                                    help='random/center crop image to square with given size during train/test(default: %(default)s)')
        data_group.add_argument('--normalize', '-norm', action='store_true', help='normalize image values with imagenet mean&std')
        
        # Model training loss
        model_group.add_argument('--beta',  metavar='%s', type=int, default=1, 
                                     help='scaling factor before the orientation loss term(default: %(default)s)')        
        model_group.add_argument('--learn_weighting',  action='store_true',  
                                     help='learn the weighting factor during training')
        model_group.add_argument('--homo_init', metavar=('%f[Sx]', '%f[Sq]'), type=float, nargs=2, default=[0.0, -3.0], 
                                     help='initial guess for homoscedastic uncertainties variables(default: %(default)s)')
        model_group.add_argument('--epochs', '-ep', metavar='%d', type=int, default=900,
                                     help='number of training epochs(default: %(default)s)')
        model_group.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'],  
                                     help='specify optimizer(default: %(default)s)')
        model_group.add_argument('--epsilon', '-eps', metavar='%f', type=float, default=1.0, 
                                     help='epsilon factor for Adam(default: %(default)s)')              
        model_group.add_argument('--momentum', '-mom', metavar='%f', type=float, default=0.9, 
                                     help='momentum factor for SGD(default: %(default)s)')      
        model_group.add_argument('--lr_init', '-lr', metavar='%f', type=float, default=5e-3,
                                     help='initial learning rate(default: %(default)s)') 
        model_group.add_argument('--lr_decay', '-lrd', metavar=('%f[decay factor]', '%d[step size]'), nargs=2, default=None,
                                     help='learning rate decay factor and step(default: %(default)s)')
        model_group.add_argument('--weight_decay', '-wd', metavar='%f', type=float, default=1e-4, 
                                     help='weight decay rate(default: %(default)s)')        

        # Visdom server setting for visualization
        visdom_group.add_argument('--visenv', '-venv', metavar='%s', type=str, default=None, 
                                      help='the environment for visdom to save all data(default: %(default)s)')
        visdom_group.add_argument('--viswin', '-vwin', metavar='%s', type=str, default=None, 
                                      help='the prefix appended to window title(default: %(default)s)')
        visdom_group.add_argument('--visport', '-vp', type=int, default=9333, 
                                      help='the port where the visdom server is running(default: %(default)s)')
        visdom_group.add_argument('--vishost', '-vh', type=str, default='localhost', 
                                      help='the hostname where the visdom server is running(default: %(default)s)')
    
        model_group.add_argument('--network', type=str,  choices=['PoseNet', 'PoseLSTM'], default='PoseNet', 
                                     help='network architecture to use(default: %(default)s)')
        
    def parse(self):
        config = self.parser.parse_args()
        return config

if __name__ == '__main__':
    conf = AbsPoseConfig().parse()

