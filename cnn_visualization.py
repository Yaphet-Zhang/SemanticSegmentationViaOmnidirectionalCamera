# -*- coding: utf-8 -*-
# @Time    : 2021-10-11 22:00
# @Author  : zhangbowen


import torch
from model.unet import UNet_VISUAL
import matplotlib.pyplot as plt
from PIL import Image
from utils.hyperparameters import NAME_NET, EPOCH
from torchvision import transforms
from utils.hyperparameters import INPUT_SIZE
import warnings
warnings.filterwarnings('ignore')


########## visualize hidden weight ##########
def visualize_hidden_weight(images, num_sample, num_channel):
    plt.figure('visualization')
    plt.title('Visualize hidden weight')
    for i in range(num_sample):
        image = images[i] # [N, C, H, W] >> [C, H, W]
        image = image.numpy() # tensor >> ndarray
        image = image / 2 + 0.5  # delete normalization
        for j in range(num_channel):
            plt.subplot(num_sample, num_channel, i*num_channel+j+1)
            plt.imshow(image[j], cmap='gray')
            plt.axis('off')
    plt.savefig('visualize_hidden_weight.png')
    plt.show()


########## visualize hidden output ##########
def visualize_hidden_output(images, num_channel):    
    plt.figure('visualization')
    plt.title('Visualize hidden output')
    images = images[0] # [N, C, H, W] >> [C, H, W]
    images = images.numpy() # tensor >> ndarray
    # images = images / 2 + 0.5  # delete normalization
    for j in range(2): # if show all: set num_channel roops
        plt.subplot(1, 2, j+1) # hidden layer num_channels = a*b(need manually)
        image = images[j]
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig('out.png')
    plt.show()


if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = UNet_VISUAL(in_ch=3 , out_ch=2).to(device)
    weights_path = 'weights/' + NAME_NET + str(EPOCH) + '.pth'
    net.load_state_dict(torch.load(weights_path, map_location=device))


    net.eval()
    with torch.no_grad():        
        '''
        ########## check visualize hidden weight ##########
        # weight1
        weight1 = net.conv1[0].weight # [64, 3, 9, 9]
        weight1 = weight1.to('cpu')
        num_sample1 = weight1.shape[0]
        num_channel1 = weight1.shape[1]

        # weight2
        weight2 = net.conv2[0].weight # [32, 64, 1, 1]
        weight2 = weight2.to('cpu')
        num_sample2 = weight2.shape[0]
        num_channel2 = weight2.shape[1]

        # weight3
        weight3 = net.conv3[0].weight # [3, 32, 5, 5]
        weight3 = weight3.to('cpu')
        num_sample3 = weight3.shape[0]
        num_channel3 = weight3.shape[1]

        visualize_hidden_weight(images=weight3, num_sample=3, num_channel=3)
        '''


        ########## check visualize hidden output ##########
        image_file_path = 'data/2.jpg'  ##  parameters ###
        image = Image.open(image_file_path) # PIL: [H W C]
        # image = transforms.ToTensor()(img) # tensor: [C H W]
        transform = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor(),
        ]) 
        image = transform(image) # [C, H, W]
        image = image.unsqueeze(0).to(device) # [N, C, H, W]

        # need to manually check hidden output shape
        output = net(image).to('cpu')
        num_channel = output.shape[1]

        visualize_hidden_output(images=output, num_channel=num_channel)
