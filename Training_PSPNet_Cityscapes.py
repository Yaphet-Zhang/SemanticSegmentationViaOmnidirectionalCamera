#!/usr/bin/env python
# coding: utf-8
import os
import random
import math
import time
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.miou import mIoU
import pandas as pd
from model.pspnet import PSPNet
from tqdm import tqdm
from utils.hyperparameters import NUM_CLASSES, BATCH_SIZE, EPOCH, INPUT_SIZE, NAME_NET




########## fix random number ##########
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)




########## make train/val list ##########
# create VOCzbw/ImageSetsSegmentation directory
flag=os.path.exists(r'./data/Cityscapes/VOCzbw/ImageSets')
if not flag :
    os.makedirs(r'./data/Cityscapes/VOCzbw/ImageSets/Segmentation')

# read img path list
jpg_path_list=os.listdir('./data/Cityscapes/VOCzbw/JPEGImages')

# write img path in text
train_txt_obj=open(r'./data/Cityscapes/VOCzbw/ImageSets/Segmentation/train.txt','w')
val_txt_obj=open(r'./data/Cityscapes/VOCzbw/ImageSets/Segmentation/val.txt','w')

for jpg_path in jpg_path_list :
    if jpg_path[:9]=='frankfurt' or jpg_path[:6]=='lindau' or jpg_path[:7]=='munster':
        val_txt_obj.write(jpg_path[:-4]+'\n') # write val.txt
    else:
        train_txt_obj.write(jpg_path[:-4]+'\n') # write train.txt
train_txt_obj.close()
val_txt_obj.close()




########## make path list ##########
from utils.dataloader_Cityscapes import make_datapath_list, DataTransform, VOCDataset
rootpath = './data/Cityscapes/VOCzbw/'  #####  parameters  #####
train_img_list, train_anno_list, val_img_list, val_anno_list, train_mask_list = make_datapath_list(rootpath=rootpath)




########## Dataset ##########
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
train_dataset=VOCDataset(
    img_list=train_img_list, 
    anno_list=train_anno_list,
    mask_list=False, 
    phase='train', 
    transform=DataTransform(
        input_size=INPUT_SIZE, 
        color_mean=color_mean, 
        color_std=color_std
    )
)
val_dataset = VOCDataset(
    img_list=val_img_list, 
    anno_list=val_anno_list, 
    mask_list=False,
    phase='val', 
    transform=DataTransform(
        input_size=INPUT_SIZE, 
        color_mean=color_mean, 
        color_std=color_std
    )
)




########## DataLoader ##########
train_dataloader = data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
val_dataloader = data.DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

# Datalodar -> dictionary
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}




########## network ##########
##### from scratch
# change the output numbers of the 'last Convolution Classification Layer'
net = PSPNet(n_classes=NUM_CLASSES)
net.decode_feature.classification = nn.Conv2d(
    # 1: 512, 256; 0.5: 256, 128; 0.25: 128, 64.
    in_channels=512, out_channels=NUM_CLASSES, kernel_size=1, stride=1, padding=0)
net.aux.classification = nn.Conv2d(
    in_channels=256, out_channels=NUM_CLASSES, kernel_size=1, stride=1, padding=0)

# initial the PSPNet_new by Xavier Initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # if bias exists
            nn.init.constant_(m.bias, 0.0)
net.apply(weights_init)
print('from scratch weight initialization is loaded')
print('====================')




########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print('total gpu: {}'.format(torch.cuda.device_count()))
    print('gpu name: {}'.format(torch.cuda.get_device_name()))
    print('memory per gpu: {:.2f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))

if torch.cuda.device_count() > 1:
    # refer to: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    net = nn.DataParallel(net) # 1-node, multi-GPU
    print('{} gpu will be used'.format(torch.cuda.device_count()))
else:
    print('{} gpu will be used'.format(1))

net.to(device)
print('====================')




########## loss function ##########
aux_weight=0.4
class PSPLoss(nn.Module):
    def __init__(self, aux_weight):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight


    def forward(self, outputs, labels):
        '''
        outputs[0]: [N, C, H, W], torch.float32
        outputs[1]: [N, C, H, W], torch.float32
        labels: [N, H, W], torch.int64
        '''
        ### primary loss
        loss = F.cross_entropy(outputs[0], labels, reduction='mean') # include softmax + one-hot encoding
        ### auxiliary loss
        loss_aux = F.cross_entropy(outputs[1], labels, reduction='mean') # include softmax + one-hot encoding
        ### final loss
        loss_final = loss + self.aux_weight * loss_aux

        return loss_final

criterion = PSPLoss(aux_weight=aux_weight)




########## Optimizer(Adam) ##########
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0.0001)




# set a scheduler for slowing down the learning rate per epoch
def lambda_epoch(epoch):
    max_epoch = 500
    return math.pow((1-epoch/max_epoch), 0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)




########## Training & Validation ##########
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs, mIoU):

    # speed up the network
    torch.backends.cudnn.benchmark = True
    # how many images
    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size
    # iteration counter
    iteration = 1
    logs = []
    # multiple minibatch
    batch_multiplier = 3         
    YY_train=[]
    YY_val=[]

    # initialize loss & acc & miou list (epochs)
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_miou_list = []
    val_miou_list = []
    for epoch in range(num_epochs):        
        # initialize acc & miou (1 epoch)
        train_acc = 0
        val_acc = 0
        train_miou = 0
        val_miou = 0

        # save the start time
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  
        epoch_val_loss = 0.0  
        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # loop of training & validation per epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # training mode
                scheduler.step()  # update scheduler
                optimizer.zero_grad()
                print('（train）')
            else:
                if True:
                # if((epoch+1) % 5 == 0):
                    net.eval()   # validation mode
                    print('-------------')
                    print('（val）')
                else:
                    continue
            # use data per minibatch
            count = 0  # multiple minibatch
            for imges, anno_class_imges, mask_0_or_1 in tqdm(dataloaders_dict[phase]):
                if imges.size()[0] == 1:
                    continue

                # image to device
                imges = imges.to(device)
                # labels to device
                anno_class_imges = anno_class_imges.to(device)  

                # update the parameters
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):

                    # imges: [N, 3, 475, 475], pix, float32
                    # outputs[0]/[1]: [N, C, 475, 475], prob, float32
                    outputs = net(imges) 

                    # criterion = F.cross_entropy(outputs[0] + aux_weight * outputs[1], labels, reduction='mean') 
                    # labels: [N, 475, 475], e.g. 0, 1, 2, uint8 
                    # labels.long(): [N, 475, 475], e.g. 0, 1, 2, int64
                    # mask_0_or_1: [N, 475, 475], unit8
                    # mask_0_or_1: [N, 475, 475], int64
                    # loss: scalar
                    labels = anno_class_imges.long()
                    loss = criterion(outputs, labels) / batch_multiplier 

                    # predicts/labels: [N, 475, 475] e.g. 0, 1, 2 int64
                    predicts = torch.argmax(outputs[0], dim=1) 

                    # sum acc (all iterations)
                    if phase == 'train':
                        train_acc += torch.sum(predicts == labels).item() / len(labels) / (labels.shape[1]*labels.shape[2])
                    elif phase == 'val':
                        val_acc += torch.sum(predicts == labels).item() / len(labels) / (labels.shape[1]*labels.shape[2])

                    # sum miou (all iterations)
                    if phase == 'train':
                        train_miou += mIoU(input=predicts.to('cpu'), target=labels.to('cpu'), classNum=NUM_CLASSES)
                    elif phase == 'val':
                        val_miou += mIoU(input=predicts.to('cpu'), target=labels.to('cpu'), classNum=NUM_CLASSES)

                    # back propagation
                    if phase == 'train':
                        loss.backward()  # calculate the gradient
                        count -= 1  # multiple minibatch
                        if (iteration % 10 == 0):  
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier
       
            # mean acc (1 epoch)
            if phase == 'train':
                epoch_train_acc = train_acc / len(dataloaders_dict[phase])
            elif phase == 'val':
                epoch_val_acc = val_acc / len(dataloaders_dict[phase])

            # mean miou (1 epoch)
            if phase == 'train':
                epoch_train_miou = train_miou / len(dataloaders_dict[phase])
            elif phase == 'val':
                epoch_val_miou = val_miou / len(dataloaders_dict[phase])

        # loss list
        train_loss_list.append(epoch_train_loss/num_train_imgs)
        val_loss_list.append(epoch_val_loss/num_val_imgs)   

        # acc list (epochs)
        train_acc_list.append(epoch_train_acc)
        val_acc_list.append(epoch_val_acc)

        # acc list (epochs)
        train_miou_list.append(epoch_train_miou)
        val_miou_list.append(epoch_val_miou)

        # per epoch 
        t_epoch_finish = time.time()
        print('-------------')
        # loss
        print('epoch {} || Train loss: {:.4f} || Val loss: {:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        # acc
        print('epoch {} || Train acc: {:.4f} || Val acc: {:.4f}'.format(epoch+1, epoch_train_acc, epoch_val_acc))
        # miou
        print('epoch {} || Train miou: {:.4f} || Val miou: {:.4f}'.format(epoch+1, epoch_train_miou, epoch_val_miou))
        # time
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()        
     
        ########## save log ##########
        log_epoch = {'epoch':epoch+1, 
                     'train_loss':epoch_train_loss/num_train_imgs, 
                     'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        log_path = 'weights/' + NAME_NET + '_' + str(EPOCH) + '_Cityscapes' + '.csv'
        df.to_csv(log_path)

    ########## save weights ##########
    weights_path = 'weights/' + NAME_NET + '_' + str(EPOCH) + '_Cityscapes' + '.pth'
    torch.save(net.to(device).state_dict(), weights_path)
    print(weights_path + ': saved!')
    print('====================')

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, train_miou_list, val_miou_list 


train_loss_list, val_loss_list, train_acc_list, val_acc_list, train_miou_list, val_miou_list = train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=EPOCH, mIoU=mIoU)




########## visualization ##########
# visualize loss
plt.figure('loss')
plt.title('Train loss & Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim((0, 0.5))
# plt.yticks(np.arange(0, 0.5, 0.03))
# train loss
plt.plot(range(1, EPOCH+1), train_loss_list, 'b-', label='Train loss')
# val loss
plt.plot(range(1, EPOCH+1), val_loss_list, 'r-', label='Val loss')
plt.legend()
plt.savefig('weights/' + NAME_NET + '_' + str(EPOCH) + '_loss_Cityscapes' + '.png')
# plt.show()


# visualize acc
plt.figure('acc')
plt.title('Train acc & Val acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
# train acc
plt.plot(range(1, EPOCH+1), train_acc_list, 'b-', label='Train acc')
# val acc
plt.plot(range(1, EPOCH+1), val_acc_list, 'r-', label='Val acc')
# plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.legend()
plt.savefig('weights/' + NAME_NET + '_' + str(EPOCH) + '_acc_Cityscapes' + '.png')
# plt.show()


# visualize miou
plt.figure('miou')
plt.title('Train miou & Val miou')
plt.xlabel('Epoch')
plt.ylabel('Miou')
# train miou
plt.plot(range(1, EPOCH+1), train_miou_list, 'b-', label='Train miou')
# val miou
plt.plot(range(1, EPOCH+1), val_miou_list, 'r-', label='Val miou')
# plt.yticks(np.arange(0.0, 0.51, 0.05))
plt.legend()
plt.savefig('weights/' + NAME_NET + '_' + str(EPOCH) + '_miou_Cityscapes' + '.png')
# plt.show()