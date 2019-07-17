from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from config import batch_size,image_crop_size

'''
transforms 用于转换图片
DataLoader 用于创建实例，保存图片
'''


# 定义train set的转换，随机翻转，裁剪，正则化
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(image_crop_size,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# 加载train set(使用上面的方法transform）
train_set = CIFAR10(root="./data",train=True,transform=train_transformations,download=True)  # 设定根目录

# 为train_set 创建加载程序
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2) # shuffle：洗牌？


# 定义test set 的转换，无需裁剪，旋转
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集，注意这里的train设为false
test_set = CIFAR10(root="./data", train=False, transform=test_transformations, download=True)

# 为测试集创建加载程序，注意这里的shuffle设为false
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

