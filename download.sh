#!/bin/bash
sudo apt install pip
pip3 install visdom
pip3 install pytorch
pip3 install torchvision
pip3 install visdom
pip3 install jupyter
pip3 install matplotlib
pip3 install transforms3d

mkdir data
wget -P './data' 'https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip'