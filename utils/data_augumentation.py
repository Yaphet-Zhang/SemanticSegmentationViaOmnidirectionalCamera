# 注意　アノテーション画像はカラーパレット形式（インデックスカラー画像）となっている。
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np




class Compose(object):
    '''
    preprocess according to DataTransform class's order
    preprocess img, label, mask, simultaneously 
    '''
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, img, anno_class_img, mask):
        for tran in self.transforms:
            img, anno_class_img, mask = tran(img, anno_class_img, mask)

        return img, anno_class_img, mask




class Scale(object):
    '''
    size is not changed
    if scale > 1:
        random position crop
    if scale < 1:
        padding black
    '''
    def __init__(self, scale):
        self.scale = scale


    def __call__(self, img, anno_class_img, mask):
        # img.size: [W, H]
        width = img.size[0]
        height = img.size[1]

        # random scale factor
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        # resize img
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)
        # resize label
        anno_class_img = anno_class_img.resize((scaled_w, scaled_h), Image.NEAREST)
        # resize mask
        if mask:  
            mask = mask.resize((scaled_w, scaled_h), Image.NEAREST)

        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))
            top = scaled_h-height
            top = int(np.random.uniform(0, top))
            # crop
            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop((left, top, left+width, top+height))
            if mask:  
                mask = mask.crop((left, top, left+width, top+height))
        else:
            p_palette = anno_class_img.copy().getpalette()
            if mask: 
                m_palette = mask.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()
            if mask:  
                mask_original = mask.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))
            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            # padding
            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))
            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original, (pad_width_left, pad_height_top))
            if mask:  
                mask = Image.new(mask.mode, (width, height), (0))
                mask.paste(mask_original, (pad_width_left, pad_height_top))

            anno_class_img.putpalette(p_palette)
            if mask:  
                mask.putpalette(m_palette)

        return img, anno_class_img, mask




class RandomRotation(object):
    '''
    
    '''
    def __init__(self, angle):
        self.angle = angle


    def __call__(self, img, anno_class_img, mask):
        # random rotation factor
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        # rotate
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)
        if mask:  
            mask = mask.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img, mask




class RandomMirror(object):
    '''
    horizontal flip with 50% probability  
    '''
    def __call__(self, img, anno_class_img, mask):
        if np.random.randint(2): # 0 or 1
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
            if mask:
                mask = ImageOps.mirror(mask)

        return img, anno_class_img, mask




class Resize(object):
    '''
    resize to input_size
    '''
    def __init__(self, input_size):
        self.input_size = input_size


    def __call__(self, img, anno_class_img, mask):
        # resize
        img = img.resize((self.input_size, self.input_size), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST)
        if mask:  
            mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img, mask




class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std


    def __call__(self, img, anno_class_img, mask):

        # img
        # img.save('./data/iiiiiiiiiimg.png')
        img = transforms.functional.to_tensor(img) # PIL -> tensor, normalization: [0, 255] -> [0, 1]
        img = transforms.functional.normalize(img, self.color_mean, self.color_std) # standardization: mean, std

        # label
        # anno_class_img.save('./data/llllllllllabel.png')
        anno_class_img = np.array(anno_class_img) # PIL -> ndarray
        anno_class_img[np.where(anno_class_img == 255)] = 0 # 255 of ambiguous -> 0 (background)
        anno_class_img = torch.from_numpy(anno_class_img) # ndarray -> tensor

        # mask
        if mask:
            # mask.save('./data/mmmmmmmmmmask.png')
            mask = np.array(mask) # PIL -> ndarray
            
            # 0 or 255 --> 0 or 1 (for: 0 * loss, where is the position of mask) 
            no_mask_position = np.where(mask == 255)
            mask[no_mask_position] = 1 # 255 -> 1 (no mask)
            mask_position = np.where(mask == 0)
            mask[mask_position] = 0 # 0 -> 0 (mask)

            mask_0_or_1 = mask
            mask_0_or_1 = torch.from_numpy(mask_0_or_1) # ndarray -> tensor
        else:
            mask_0_or_1 = mask

        return img, anno_class_img, mask_0_or_1

