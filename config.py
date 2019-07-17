'''
config for super parameters
'''

# config

# super parameter
learn_rate = 0.001
weight_decay = 0.0001
epoch = 200
batch_size = 48
class_num = 10
momentum = 0.9  # SGD momentum

# image data transform
image_crop_size = 32 # 裁剪大小 x*x
img_size = 32

# load and save check point
load_checkpoint = True
#load_ckpt_path = './checkpoint/ckpt.pth'
load_ckpt_path = './checkpoint/acc_87_batch_s_48_ckpt.pth'