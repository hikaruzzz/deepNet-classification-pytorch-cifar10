'''
customs normalize model /46
image show model /
'''


from backbones import *
from config import *
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt



def load_checkpoint():
    # Load checkpoint.
    print('>>> Resuming from checkpoint')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if torch.cuda.is_available():
        checkpoint = torch.load(load_ckpt_path)
    else:
        checkpoint = torch.load(load_ckpt_path,map_location='cpu')

    model_1.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('>>> Resume success with checkpoint from:',load_ckpt_path)

def transfrom_img(file_name):
    # load_images

    #torch.rand(10,3,32,32)
    img1 = Image.open(file_name)
    assert np.array(img1).shape[2] <= 3,"!! {} not RGB format,channel > 3"

    img2 = img1.resize((img_size,img_size),resample=Image.HAMMING)

    img_Tensor = torch.tensor(np.transpose(img2,[2,0,1])[np.newaxis,:,:,:],dtype=torch.float32) # !!! xx()(xx)

    # Normilizer
    # 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean
    img_Tensor = img_Tensor.float().div(255)
    img_Tensor = img_Tensor.sub_(0.5).div_(0.5)

    return np.array(img1), img2, img_Tensor


if __name__ == "__main__":
    cuda_isavail = torch.cuda.is_available()  # 查看cuda可用
    print(">>> is cuda availavle = ", cuda_isavail)

    # create model
    if model_name == "xjbNet":
        model_1 = XJBNet(class_n=class_num)
    elif model_name == "ResNet50":
        model_1 = ResNet50()
    elif model_name == "MobileNetV2":
        model_1 = MobileNetV2(num_classes=class_num)
    elif model_name == "ResNet101":
        model_1 = ResNet101()
    elif model_name == "DPN":
        model_1 = DPN92()
    else:
        assert 0, print("not found model")

    if cuda_isavail:
        model_1.cuda()
    else:
        model_1.cpu()

    model_1.eval()  # 切换eval模式

    # load checkpoint
    load_checkpoint()

    # load images
    demo_pth = os.path.join('demo')
    plt.figure(1)
    for idx,f_name in zip(range(100),os.listdir(demo_pth)):
        img, img_resize, img_Tensor = transfrom_img(os.path.join(demo_pth,f_name))
        plt.subplot(1,2,1)
        plt.title("origin image")
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(np.reshape(img_resize,[32,32,3]))

        # inference
        if cuda_isavail:
            img_Tensor = img_Tensor.to('cuda')
        else:
            img_Tensor = img_Tensor.to('cpu')

        outputs = model_1(img_Tensor)
        _, pred = torch.max(outputs.data, 1)

        # show
        classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
        print("image:{}, class pred:{}".format(f_name,classes[pred.tolist()[0]]))
        #print("confidence: ",outputs.data)
        plt.title("resize image predict:\n\n"+classes[pred.tolist()[0]])
        plt.show()
