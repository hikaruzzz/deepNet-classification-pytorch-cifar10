from load_data import *
from config import *
from torch.autograd import Variable
from backbones import *
import os
import time
import torch.optim

#from tensorboardX import SummaryWriter

'''
使用SGD optimizer
'''


# saver函数
def save_models(epoch,acc,model_name):
    state = {
        'net': model_1.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    checkpoint_path = os.path.join("checkpoint")
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    save_path = os.path.join(checkpoint_path, '{}_acc_{}_batch_s_{}_ckpt.pth'.format(model_name,acc, batch_size))
    torch.save(state,save_path)
    print("checkpoint saved in ./{}".format(save_path))


def test():
    model_1.eval() # 切换eval模式
    test_acc = 0.0
    i=0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):  # len(test_loader)=batch_size???
            # 一个batch_size给的，最终会把整个数据集跑一遍

            if cuda_isavail:
                images = Variable(images.cuda()) # 每一步把图像和标签移往GPU，在Variable中将它们封装
                labels = Variable(labels.cuda())

            # Predict classes from the test set
            outputs = model_1(images)
            _, pred = torch.max(outputs.data, 1) # 选择最大预测值，max_soft？？
            total += labels.size(0)
            test_acc += torch.sum(pred == labels.data)

        # Compute the average acc and loss over all 10000 test images
        test_acc = 100. * test_acc / total

        return test_acc


def train(epochs):

    if load_checkpoint:
        # Load checkpoint.
        print('>>> Resuming from checkpoint')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        if torch.cuda.is_available():
            checkpoint = torch.load(load_ckpt_path)
        else:
            checkpoint = torch.load(load_ckpt_path, map_location='cpu')

        model_1.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print('>>> Resume success')
    # define optimizer & Loss
    # optimizer = Adam(model_1.parameters(),lr=learn_rate,weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model_1.parameters(), lr=learn_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()  # Entropy Loss

    print(">>> batch_size= ",batch_size)
    best_acc = 0.0
    for epoch in range(epochs):
        model_1.train() # 切换train模式
        train_acc = 0.0
        train_loss = 0.0
        total = 0
        time_s = time.time()
        i=0
        for i,(images,labels) in enumerate(train_loader): # 每次读出一个batch size的数据,len(train_loader)=total/batchsize

            if i % 100==0:
                print("calc over step: {},time used: {:.0f} ms".format(100,(time.time() - time_s)*1000))
                time_s = time.time()

            if cuda_isavail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())


            # 清除所有累积梯度
            optimizer.zero_grad()
            # 用来自测试集的图像预测类
            outputs = model_1(images)
            # 根据实际标签和预测值计算损失
            loss = criterion(outputs, labels)
            # 后传，计算梯度？
            loss.backward()
            # 按梯度调整参数
            optimizer.step()

            # train_loss,每step都叠加一次，所以对于每个step的真实loss要 / 相应的step数（i+1）
            train_loss += loss.cpu().item()  # 直接从 cuda 中取数据使用的方法,为什么要 *上 size
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print("x",total,"len(train_loader): ",len(train_loader)) # 最后一个step的batch_size != 正常的batchsize(不重复）
            train_acc += torch.sum(pred == labels.data)  # 所有批次中的正确预测值相加

        # 利用optimizer.para_groups["lr"]来动态改变learn rate ，这是Adam optimizer专用
        #lr_adj = lr_adjust.adjust_learn_rate(epoch=epoch, lr=learn_rate)
        #optimizer.param_group["lr"] = lr_adj
        # 计算模型在50000张(实际batch_size*len(train_loader))训练图像上的准确率和损失值
        # len（train_loader)*batch_size=总的step数,最后一个step的batch_size不一样
        #print("len(train_loader)*batch_size: ",len(train_loader)*batch_size,"total: ",total)
        train_acc_avg = 100. * train_acc / total
        train_loss_avg = train_loss / (i+1)

        # 用测试集评估
        test_acc = test()

        # 若测试准确率高于当前最高准确率，则保存模型
        if test_acc > best_acc:
            save_models(epoch,test_acc,model_name)
            best_acc = test_acc

        print("Epoch {}, Train Accuracy avg: {:.3f}% , TrainLoss: {:.3f} , Test Accuracy: {:.3f}%".format(epoch, train_acc_avg, train_loss_avg,test_acc))
        #writer.add_scalars('Accuracy/acc',{'train_acc':train_acc_avg,'test_acc':test_acc},epoch)
        #writer.add_scalars('Accuracy/Loss',{'train_loss':train_loss_avg},epoch)
    #writer.close()


if __name__ == "__main__":
    #writer = SummaryWriter()

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

    train(epoch)