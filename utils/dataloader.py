import os.path 
from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import torch.utils.data as data
from PIL import Image
import numpy as np




def make_datapath_list(rootpath):
    imgpath_template = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = os.path.join(rootpath, 'SegmentationClassPNG', '%s.png')
    train_id_names = os.path.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_id_names = os.path.join(rootpath, 'ImageSets/Segmentation/val.txt') 
    
    # train
    train_img_list = []
    train_anno_list = []
    train_mask_list = []
    for line in open(train_id_names):    
        file_id = line.strip()
        # img
        train_img_path = imgpath_template % file_id
        train_img_list.append(train_img_path)
        # label
        train_anno_path = annopath_template % file_id
        train_anno_list.append(train_anno_path)
        # mask
        train_mask_path = './data/VOCdevkit/VOCzbw_mask/SegmentationClassPNG/%s.png' % file_id
        train_mask_list.append(train_mask_path)

    # val
    val_img_list =[] 
    val_anno_list = []
    for line in open(val_id_names):
        file_id = line.strip()
        # img
        val_img_path = imgpath_template % file_id
        val_img_list.append(val_img_path)
        # label
        val_anno_path = annopath_template % file_id
        val_anno_list.append(val_anno_path)

    return  train_img_list, train_anno_list, val_img_list, val_anno_list, train_mask_list




class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform={
            'train': Compose([
                # Scale(scale=[0.5,1.5]),
                RandomRotation(angle=[-10,10]),
                RandomMirror(),
                Resize(input_size=input_size),
                Normalize_Tensor(color_mean=color_mean,color_std=color_std)
            ]),
            'val' : Compose([
                Resize(input_size=input_size),
                Normalize_Tensor(color_mean=color_mean,color_std=color_std)
            ])
        }


    def __call__(self, phase, img, anno_class_img, mask):
        img, anno_class_img, mask_0_or_1 = self.data_transform[phase](img, anno_class_img, mask) 
        
        return img, anno_class_img, mask_0_or_1 




class VOCDataset(data.Dataset):
    '''
    if phase = 'train': 
        img_list = train_img_list, 
        anno_list = train_anno_list 
        mask_list = train_mask_list
    elif phase = 'val': 
        img_list = val_img_list, 
        anno_list = val_anno_list   
    '''  
    def __init__(self, img_list, anno_list, mask_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.mask_list = mask_list
        self.phase = phase
        self.transform = transform
        

    def __len__(self):
        '''return numbers of jpg'''
        return len(self.img_list)
        

    def __getitem__(self, index):
        '''
        return pre-processed tensor of jpg & png
        '''
        img, anno_class_img, mask_position = self.pull_item(index)
        return img , anno_class_img, mask_position


    def pull_item(self, index):
        # read jpg img
        img = Image.open(self.img_list[index]) # [H, W, C]
        # read png label
        anno_class_img = Image.open(self.anno_list[index]) # [H, W]
        # read png mask
        if self.phase=='train':
            mask = Image.open(self.mask_list[index]) # [H, W]
        elif self.phase=='val':
            mask = False

        # pre-process jpg img & png label & png mask (by calling DataTransform.__call__)
        img, anno_class_img, mask_0_or_1 = self.transform(self.phase, img, anno_class_img, mask)
        

        return img, anno_class_img, mask_0_or_1
