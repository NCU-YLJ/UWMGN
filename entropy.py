import torch
from utils.denoising_utils import *
from numpy import array, shape, where, in1d
import math

import math

# 以整型数据为例，给出其信息熵的计算程序。
###########################################
'''统计已知数据中的不同数据及其出现次数'''


###########################################
def StatDataInf(data):
    dataArrayLen = len(data)
    diffData = [];
    diffDataNum = [];
    dataCpy = data;
    for i in range(dataArrayLen):
        count = 0;
        j = i
        if (dataCpy[j] != '/'):
            temp = dataCpy[i]
            diffData.append(temp)
            while (j < dataArrayLen):
                if (dataCpy[j] == temp):
                    count = count + 1
                    dataCpy[j] = '/'
                j = j + 1
            diffDataNum.append(count)
    return diffData, diffDataNum


###########################################
'''计算已知数据的熵'''

###########################################
def DataEntropy(data, diffData, diffDataNum):
    dataArrayLen = len(data)
    diffDataArrayLen = len(diffDataNum)
    entropyVal = 0;
    for i in range(diffDataArrayLen):
        proptyVal = diffDataNum[i] / dataArrayLen
        entropyVal = entropyVal - proptyVal * math.log2(proptyVal)
    return entropyVal


fname =Image.open('data/low_light/two/1.bmp')   # 低照度图像 512*512
img_np = pil_to_np(fname)  # 待处理图  ndarray  [1,512,512]
img_np = np.squeeze(img_np)
w, h = img_np.shape
img_np = img_np.reshape(1,w*h)
img = np.squeeze(img_np)  # 去掉第一个维度
print(img)
img = img.tolist() #转为list
[diffData, diffDataNum] = StatDataInf(img)
entropyVal = DataEntropy(img, diffData, diffDataNum)
print(entropyVal)

data = [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1]
[diffData, diffDataNum] = StatDataInf(data)
entropyVal = DataEntropy(data, diffData, diffDataNum)
print(entropyVal)

data = [1, 2, 3, 4, 2, 1, 2, 4, 3, 2, 3, 4, 1, 1, 1]
[diffData, diffDataNum] = StatDataInf(data)
entropyVal = DataEntropy(data, diffData, diffDataNum)
print(entropyVal)

data = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
[diffData, diffDataNum] = StatDataInf(data)
entropyVal = DataEntropy(data, diffData, diffDataNum)
print(entropyVal)
data = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5]
[diffData, diffDataNum] = StatDataInf(data)
entropyVal = DataEntropy(data, diffData, diffDataNum)
print(entropyVal)


# fname =Image.open('data/low_light/two/1.bmp')   # 低照度图像 512*512
# img_np = pil_to_np(fname)  # 待处理图  ndarray  [1,512,512]
# img_np = np.squeeze(img_np)
#
# # w, h = img_np.shape
#
# # data = img_np.reshape(1, w*h)
