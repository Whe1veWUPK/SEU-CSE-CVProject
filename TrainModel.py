import torch
import torch.nn as nn
import os
from Net import vgg16
from torch.utils.data import DataLoader
from DataProcesser import *

'''Train Set'''

path=os.getcwd()
trainPath = path+r'\paths\trainPicturesPaths.txt'  # storing all train set pictures'path txt path
with open(trainPath, 'r') as f:
    trainLines = f.readlines()
np.random.seed(10101)
np.random.shuffle(trainLines)  # Data scrambled
np.random.seed(None)
numOfTrain = len(trainLines)

# Input picture size
inputShape = [224, 224]
trainData = DataGenerator(trainLines[:numOfTrain], inputShape, True)


"""Load Data"""
TrainSet = DataLoader(trainData, batch_size=32)  # 训练集batch_size读取小样本，规定每次取多少样本

'''Make Net'''
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # choose the device and the cuda is preferred
net = vgg16(True, progress=True, num_classes=2)
net.to
'''Choose learning rate and optimizer'''
lr = 0.0001  # the learning rate
optim = torch.optim.Adam(net.parameters(), lr=lr)  # import the net and the learning rate
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)  # read by step length equals to 1
'''Train'''
epochs = 5  # train times
for epoch in range(epochs):
    totalLoss = 0  # total loss
    num = 1
    for data in TrainSet:
        img, label = data
        print("Now is training the data "+str(num))
        num+=1
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optim.zero_grad()
        output = net(img)
        trainLoss = nn.CrossEntropyLoss()(output, label).to(device)
        trainLoss.backward()
        optim.step()  # update the optim
        totalLoss += trainLoss  # sum the loss
    sculer.step()

    print("The loss of the train set：{}".format(totalLoss))

    torch.save(net.state_dict(),
               path+r'\trained models\CD ClassificationModel{}.pth'.format(epoch + 1))
    print("The model has been saved!")
