#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np
import torch
import argparse


### Add some argument on command prompt
parser=argparse.ArgumentParser(description="10 classes inference of PSPNet by PyTorch: ")
parser.add_argument('--start', type=int, default=1, help='number of start image which you want to predict')
parser.add_argument('--end', type=int, default=100, help='number of end iamge which you want to predict')
args=parser.parse_args()


### Load network & weights
from model.pspnet import PSPNet
##########  parameters  ##########
n_classes=10
##########  parameters  ##########
# load network
net = PSPNet(n_classes=n_classes)
state_dict = torch.load("./weights/pspnet_100.pth",  #####  parameters  #####
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)
print("you've loaded the weights of 'trained by yourself'.")


### Inference by 1 GPU (2 GPU is slow)
from utils.dataloader import make_datapath_list, DataTransform
# image pre-processing
rootpath = "./data/VOCdevkit/VOCzbw/"  #####  parameters  #####
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=rootpath)
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(
    input_size=475, color_mean=color_mean, color_std=color_std)
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)  # [H W]
p_palette = anno_class_img.getpalette()
########## use 1-GPU ##########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Your device：", device)
net=net.to(device)
########## use 1-GPU ##########
########## use Multi-GPU ##########
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count()> 1:
#    net=nn.DataParallel(net)
#print("Your device：", device)
#net.to(device)
########## use Multi-GPU ##########
for i in np.arange(args.start,args.end+1):
    ##### read original image #####
    image_file_path = "./data/VOCdevkit/inference/input/"+str(i)+".jpg"  ##  parameters ###
    img = Image.open(image_file_path)  # pillo(img): [H W C]
    img_width, img_height = img.size
    ##########  parameters  ##########
    # resolution of output mask & visual image 
    img_width=int(img_width/4)
    img_height=int(img_height/4)
    ##########  parameters  ##########
    phase = "val"
    img, anno_class_img = transform(phase, img, anno_class_img)  # torch(img): [C H W]
    # start evaluation mode
    net.eval()
    x = img.unsqueeze(0)  #!!!!! torch(img): [3,475,475] -->> torch of net_input(x): ([1,3,475,475])  !!!!!#  
    # send the input data to device(GPU or CPU)
    x=x.to(device)
    outputs = net(x)
    y = outputs[0]  # torch of net_output(y): [1,10,475,475] 
    y=y[0]  # torch(y): [1,10,475,465] -->> [10,475,475]
    # get the biggest class from PSPNet's output by 'argmax'
    y=torch.argmax(y,dim=0)
    # change to CPU for using pillow  
    y=y.cpu().numpy()
    anno_class_img = Image.fromarray(np.uint8(y))  # [475,475]
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)        
    # put palette
    anno_class_img.putpalette(p_palette)     
    #####  save mask png  #####
    #anno_class_img.save("./data/VOCdevkit/inference/output/mask"+str(i)+".png")
    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')            
    for x in range(img_width):
        for y in range(img_height):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            else:
                trans_img.putpixel((x, y), (r, g, b, 150))
    # read original image
    img = Image.open(image_file_path)   # [H W C]
    img=img.resize((img_width,img_height),Image.NEAREST)
    # composite original image & mask image
    result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    #####  save visual png  #####
    result.save("./data/VOCdevkit/inference/output/visual"+str(i)+".png")
