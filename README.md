## Absolute Camera Pose Regression for Visual Localization
This repository provides implementation of PoseNet\[Kendall2015ICCV\], PoseNet-Nobeta\[Kendall2017CVPR\] which trains PoseNet using the loss learning the weighting parameter and PoseLSTM\[Walch2017ICCV\].

### Data Preparation
1. Datasets are supposed to be placed under _data/_, e.g., _data/CambridgeLandmarks_ or _data/7Scenes_.
If you want to train it on other datasets, please make sure it has same format CambridgeLandmarks, meaning pose labels are writting in **dataset_train.txt** and **dataset_test.txt**. For more details about the pose label format, you can check CambridgeLandmarks dataset documentation.

2. Download pretrained model for PoseNet initialization
The weights are pretrained on Place dataset for place recognition and has been adapted for our PoseNet implementation. It can be downloaded by executing _weights/download.sh_.
### Training Examples
Here we show an example to train a PoseNet-Nobeta model on ShopFacade scene.
For more detailed training options run `python -m abspose -h` from the repository root directory.
````
# Example to train a PoseNet on ShopFacade
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/%your_dataset_folder%' \
       --train_txt 'dataset_train.txt' --val_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.005 -wd 0.0001 \
       --learn_weighting  --homo_init 0.0 -3.0 \
       --odir 'output/posenet/nobeta/CambridgeLandmarks/ShopFacade/lr5e-3_wd1e-4_sx0.0_sq-3.0'\
````

We use [Visdom](https://github.com/facebookresearch/visdom) server to visualize the training process.  By default, training loss, validation accuracy( translation and rotation) are plotted to one figure. One can adapt it inside the code to plot other statistics. It can turned on from training options as following.
````
# Visdom option 
  --visenv %s, -venv %s the visdom environment name
  --viswin %s, -vwin %s the title of the plot window
  --visport %d, -vp %d the port where the visdom server is running(default: 9333)
  --vishost %s, -vh %s the hostname where the visdom server is running(default: localhost)

# Append these options to the previous training command
  -vp 9333 -vh 'localhost' -venv 'PoseNet-Cambridge' -vwin 'nobeta.shop.lr5e-3_wd1e-4_sx0.0_sq-3.0'
````

### Citations
````
# PoseNet Original Version
@Inproceedings{Kendall2015ICCV,
  Title                    = {PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  Author                   = {Kendall, Alex and Grimes, Matthew and Cipolla, Roberto},
  Booktitle                = {ICCV},
  Year                     = {2015},

  Optorganization          = {IEEE},
  Optpages                 = {2938--2946}
}

# PoseNet Learn Weights
@InProceedings{Kendall2017CVPR,
author = {Kendall, Alex and Cipolla, Roberto},
title = {{Geometric Loss Functions for Camera Pose Regression With Deep Learning}},
booktitle = {CVPR},
year = {2017}
}

# PoseLSTM
@InProceedings{Walch2017ICCV,
title = {Image-Based Localization Using LSTMs for Structured Feature Correlation},
author = {Walch, Florian and Hazirbas, Caner and Leal-Taixe, Laura and Sattler, Torsten and Hilsenbeck, Sebastian and Cremers, Daniel},
booktitle = {ICCV},
year = {2017}
}
````

