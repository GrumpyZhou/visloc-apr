## Absolute Camera Pose Regression for Visual Localization

### Data Preparation
Datasets are supposed to be placed under _data/_, e.g., _data/CambridgeLandmarks_ or _data/7Scenes_.
If you want to train it on other datasets, please make sure it has same format CambridgeLandmarks, meaning pose labels are writting in **dataset_train.txt** and **dataset_test.txt**. For more details about the pose label format, you can check CambridgeLandmarks dataset documentation.

### Training Examples
For detailed training options run `python -m abspose -h` from the repository root directory.
````
# Train PoseNet on ShopFacade
python -m abspose -b 75 --train -val 10 --epoch 1000 \
       --data_root 'data/your_dataset_folder' \
       --train_txt 'dataset_train.txt' --val_txt 'dataset_test.txt' \
       --dataset 'ShopFacade' -rs 256 --crop 224 \
       --network 'PoseNet'  --pretrained 'weights/googlenet_places.extract.pth'\
       --optim 'Adam' -eps 1.0 -lr 0.005 -wd 0.0001 \
       --learn_weighting  --homo_init 0.0 -3.0 \
       --odir 'output/posenet/nobeta/CambridgeLandmarks/ShopFacade/lr5e-3_wd1e-4_sx0.0_sq-3.0'\

````

##  Training Visualization
````
   -vp 9333 -vh 'srvcremers2' -venv 'PoseNet-Cambridge' -vwin 'nobeta.shop.lr5e-3_wd1e-4_sx0.0_sq-3.0'
````
We use visdom.server to record training losses and also other parameters if you adapt the code. 
You can turn it on from training options.
