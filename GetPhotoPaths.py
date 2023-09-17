import os
from os import getcwd

import numpy as np

classes = ['cat', 'dog']
sets = ['dataset']

if __name__=='__main__':
    wd=getcwd()
    path=os.getcwd()
    for se in sets:
        
        targetFile=open(path+r'\paths\PicturesPaths.txt', 'w')

        picturesPaths=se
        types=os.listdir(picturesPaths)
        for type in types:
            if type not in classes:
                continue
            classID = classes.index(type)#输出0-1 在这里0 指的是 cat 1 指的是 dog
            photosPath = os.path.join(picturesPaths, type)
            photosName = os.listdir(photosPath)
            for photoName in photosName:
                _,postfix=os.path.splitext(photoName)#该函数用于分离文件名与拓展名
                if postfix not in['.jpg','.png','.jpeg']:
                    continue
                targetFile.write(str(classID) + ';' + '%s/%s' % (wd, os.path.join(photosPath, photoName)))
                targetFile.write('\n')
        targetFile.close()
    with open(path+r'\paths\PicturesPaths.txt','r')as f:
          pictureLines=f.readlines()

    np.random.seed(10101)
    np.random.shuffle(pictureLines)  # 数据打乱
    np.random.seed(None)
    trainNum= len(pictureLines)*0.75
    index=0
    trainFile=open(path+r'\paths\trainPicturesPaths.txt','w')
    testFile=open(path+r'\paths\testPicturesPaths.txt','w')
    while index<len(pictureLines):
          if index<trainNum :
            trainFile.write(pictureLines[index])

          else:
            testFile.write(pictureLines[index])


          index+=1;

