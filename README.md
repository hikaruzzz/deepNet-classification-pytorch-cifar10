# xjbNet
# A customs deep network of classification, implemented by pytorch, with cifar-10 dataset.
## Architecture
* about the model architecture, please reference model_units.py
## How to Run
* install requirement environment  
 python = 3.6  
 pytorch = 1.1.0  
 torchvision = 0.2.2  
 matplotlib  
 PIL  
 tensorboardX (recommand) 
 cuda (recommand)  
* launch train (if need to load checkpoint, revise the parameter in config.py)  
 (GPU) run `CUDA_VISIBLE_DEVICES=0 python train_units.py`  
 (CPU) run `python train_units.py`  
* launch demo (if need to run demo with custom images, move the custom images to the path=./demo)  
 run `python demo.py`  

## Others
* change config by config.py, include super parameters.  

## Network Predict Result
* the origin image(any size) will resize to 32x32 pixel, then input to network, and predict.
* obviously, the deep network have learned some characteristics to predict classes, such as antler shape, green frog... 
![1.png](https://github.com/hikaruzzz/deepNet-classification-pytorch-cifar10/blob/master/images/1.png)
![2.png](https://github.com/hikaruzzz/deepNet-classification-pytorch-cifar10/blob/master/images/2.png)
![3.png](https://github.com/hikaruzzz/deepNet-classification-pytorch-cifar10/blob/master/images/3.png)
