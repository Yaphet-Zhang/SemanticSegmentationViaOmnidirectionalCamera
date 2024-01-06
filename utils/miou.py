import torch
import numpy as np


def one_image_mIoU(input,target,classNum):

    # 0 matrix: [4, 2, 475, 475]
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]])#创建[b,c,h,w]大小的0矩阵
    # 0 matrix: [4, 2, 475, 475]
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])#同上

    # predicts: [4, 1, 475, 475]
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    # targets: [4, 1, 475, 475]
    target = target.unsqueeze(1)#同上

    # [4, 2, 475, 475]
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    # [4, 2, 475, 475]
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上

    batchMious = []#为该batch中每张图像存储一个miou
    # [4, 2, 475, 475]
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection

    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
        miou = np.mean(ious)#计算该图像的miou

    return miou


def mIoU(input,target,classNum):

    # 0 matrix: [4, 2, 475, 475]
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]])#创建[b,c,h,w]大小的0矩阵
    # 0 matrix: [4, 2, 475, 475]
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])#同上

    # predicts: [4, 1, 475, 475]
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    # targets: [4, 1, 475, 475]
    target = target.unsqueeze(1)#同上

    # [4, 2, 475, 475]
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    # [4, 2, 475, 475]
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上

    batchMious = []#为该batch中每张图像存储一个miou
    # [4, 2, 475, 475]
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection

    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    batch_mean_miou = np.array(batchMious).mean() # 批图片的miou
   
    return batch_mean_miou


if __name__=='__main__':

    prediction=torch.rand([4, 475, 475]).long()
    target=torch.rand([4, 475, 475]).long()
    classNum=2

    miou = mIoU(input=prediction, target=target, classNum=classNum)
    print(miou)


