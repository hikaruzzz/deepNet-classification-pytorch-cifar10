# xjbNet
# A customs deep network of classification, implemented by pytorch, with cifar-10 dataset.
## Architecture
* reference model_units.py
## Network Effect
![1.png]("xxx")
![2.png]("xxx")
![3.png]("xxx")
## How to Run
* requirement environment  
 python = 3.6  
 pytorch = 1.1.0  
 torchvision = 0.2.2  
 matplotlib  
 PIL  
 tensorboardX (recommand) 
 cuda (recommand)  
* for train (if need to load checkpoint,revise the config file)  
 (GPU) run `CUDA_VISIBLE_DEVICES=0 python train_units.py`  
 (CPU) run `python train_units.py`  
* for demo (move the demo images to the path=./demo)  
 run `python demo.py`  

## Others
* change config by config.py, include super parameters.  
