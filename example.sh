#!/bin/bash

source activate visloc_apr

# Train PoseNet with Beta loss
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_train.txt' --val_pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.005 -wd 0.0001 \
       --beta 100 \
       --odir 'output/posenet/test' 
       
# Uncomment following lines for visdom usage
#        -vp 9333 -vh 'localhost' -venv 'PoseNet-Cambridge' -vwin 'nobeta.shop.lr5e-3_wd1e-4_sx0.0_sq-3.0'


# Train PoseNet with learn weight loss
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_train.txt' --val_pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.005 -wd 0.0001 \
       --learn_weighting  --homo_init 0.0 -3.0 \
       --odir 'output/posenet/test' 
       

# Train PoseLSTM  with learn weight loss
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_train.txt' --val_pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseLSTM'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.0005 -wd 0.0001 \
       --learn_weighting  --homo_init 0.0 -3.0 \
       --odir 'output/poselstm/test'


# Test a model
python -m abspose -b 75 --test \
       --data_root 'data/CambridgeLandmarks' \
       --pose_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'\
       --learn_weight \
       --resume 'output/model_exports/models/posenet/nobeta/CambridgeLandmarks/ShopFacade/lr5e-3_wd1e-4_sx0.0_sq-3.0/checkpoint_350_0.98m_6.75deg.pth' \
       --odir 'output/posenet/test' 
