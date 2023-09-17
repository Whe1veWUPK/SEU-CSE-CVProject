import torch
import os
from Net import vgg16
from DataProcesser import *
from torchvision import transforms
import torch.nn.functional as F

path=os.getcwd()
trainSetPath = path+r'\paths\trainPicturesPaths.txt'
with open(trainSetPath, 'r') as f:
    trainLines = f.readlines()
np.random.seed(10101)
np.random.shuffle(trainLines)  # shufftle data
np.random.seed(None)
'''Loading network'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device and the cuda is preferred
net = vgg16()  # input network
model = torch.load(path+r"\trained models\CD ClassificationModel5.pth",
                   map_location=device)  # load the model that has been trained, you can change it 
net.load_state_dict(model)  # import model

numOfTrain = len(trainLines)
numOfCorrect = 0
print("The num of train set pictures is" + str(numOfTrain))
classes = ['cat', 'dog']
num = 1
for line in trainLines:
    line = line[:-1]
    imagePath = line[2:]
    test = Image.open(imagePath)
    '''Process picture'''
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(test)

    net.eval()  # set predict mode
    image = torch.reshape(image, (1, 3, 224, 224))
    with torch.no_grad():
        out = net(image)
    out = F.softmax(out, dim=1)  # softmax function to definite the range
    out = out.data.cpu().numpy()

    a = int(out.argmax(1))  # the max value's index
    print("Now is identifying the picture " + str(num))
    num += 1
    if line.find(classes[a]) != -1:
        numOfCorrect += 1
        print("identify correctly")
    else:
        print("identify false")
        continue

accuracy = numOfCorrect / numOfTrain
finalAccuracy = "%.17f%%" % (accuracy * 100)
print("The accuracy of train set is " + finalAccuracy)
