## Absolute Camera Pose Regression for Visual Localization
This repository provides implementation of PoseNet\[Kendall2015ICCV\], PoseNet-Nobeta\[Kendall2017CVPR\] which trains PoseNet using the loss learning the weighting parameter and PoseLSTM\[Walch2017ICCV\].
To use our code, first download the repository:
````
git clone git@github.com:GrumpyZhou/visloc-apr.git
````

### Setup Running Environment
We tested the code on Linux Ubuntu 16.04.6 with 
````
Python 3.7
Pytorch 1.0
CUDA 8.0
````
We recommend to use *Anaconda* to manage packages. Run following lines to automatically setup a ready environment for our code.
````
conda env create -f environment.yml
conda activte visloc_apr
````
Otherwise, one can try to download all required packages seperately according to their offical documentation.
*Comments:*_The code has also been tested with Python 3.5, Pytorch 0.4, but now we have upgraded to latest versions._
### Prepare Datasets 
Our code is flexible for evaluation on various localization datasets. We use Cambridge Landmarks dataset as an example to show how to prepare a dataset:
1. Create data/ folder (optional)
````
cd visloc-apr/
mkdir data
cd data/
````
2. Download [Cambridge Landmarks Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) under _data/_  
3. Visualize the data using [notebooks/data_loading.ipynb](notebooks/data_loading.ipynb) .
4. (Optional) One can also resize the dataset images so that shorter side has 256 pixels at once makes training faster.
#### Other Datasets
If you want to train it on other datasets, please make sure it has same folder structure as Cambridge Landmarks dataset: 
````
data/target_dataset/
-- dataset_scene/
---- dataset_train.txt
---- dataset_test.txt
````
Here, **dataset_train.txt** and **dataset_test.txt** are the pose label files. For more details about the pose label format, you can check the documentation of Cambridge Landmarks dataset.


### Training
We recommend to download **pretrained** model for PoseNet initialization. The weights are pretrained on [Place](https://github.com/CSAILVision/places365) dataset for place recognition and has been adapted for our PoseNet implementation. It can be downloaded by executing [weights/download.sh](weights/download.sh).
Use [abspose.py](abspose.py) for either training or testing. For detailed training options run `python -m abspose -h` from the repository root directory.
#### Training Examples
Here we show an example to train a PoseNet-Nobeta model on ShopFacade scene.
````
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_train.txt' --val_pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.005 -wd 0.0001 \
       --learn_weighting  --homo_init 0.0 -3.0 \  
       --odir %output_dir%\
````
See more training examples in [example.sh](example.sh).
#### Training Visualization (optional)
We use [Visdom](https://github.com/facebookresearch/visdom) server to visualize the training process.  By default, training loss, validation accuracy( translation and rotation) are plotted to one figure. One can adapt it inside [utils/common/visdom_templates.py](utils/common/visdom_templates.py) to plot other statistics. It can turned on from training options as following.
````
# Visdom option 
  --visenv %s, -venv %s the visdom environment name
  --viswin %s, -vwin %s the title of the plot window
  --visport %d, -vp %d the port where the visdom server is running(default: 9333)
  --vishost %s, -vh %s the hostname where the visdom server is running(default: localhost)

# Append these options to the previous training command
  -vp 9333 -vh 'localhost' -venv 'PoseNet-Cambridge' -vwin 'nobeta.shop.lr5e-3_wd1e-4_sx0.0_sq-3.0'
````

### Testing
#### Trained models
We provide some pretrained models [here](https://vision.in.tum.de/webshare/u/zhouq/visloc-apr/models/). One can also see how the output of the program there. **However**, we did not spend much effort in tuning trainnig parameters to improve the localization accuracy, since it is not essential for us.

1. Test a model using [abspose.py](abspose.py) :
````
python -m abspose -b 75 --test \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'\
       --resume %checkpoint_path% 
       --odir %result_output_dir%
````
2. Test using [notebooks/evaluate_posenet.ipynb](notebooks/evaluate_posenet.ipynb):
We also provide notebook for evaluation which could be usful for further experiments with pretrained models.

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

