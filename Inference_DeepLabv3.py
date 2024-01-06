from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.hyperparameters import NUM_CLASSES, EPOCH, INPUT_SIZE, NAME_NET
from model.deeplabv3 import DeeplabV3
from utils.dataloader import make_datapath_list, DataTransform


########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print('total gpu(s): {}'.format(torch.cuda.device_count()))
    print('gpu name(s): {}'.format(torch.cuda.get_device_name()))
    print('gpu spec(s): {} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
print('device: {}'.format(device))
print('====================')


########## network ##########
net = DeeplabV3.to(device)
print(net)
print('====================')


########## load weights ##########
weights_path = 'weights/' + NAME_NET + str(EPOCH) + '.pth'
flag = net.load_state_dict(torch.load(weights_path, map_location=device))
print(flag)
print(weights_path + ': loaded!')
print('====================')


rootpath = "./data/VOCdevkit/VOCzbw/"  #####  parameters  #####
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=rootpath)

# show original image 
image_file_path = 'data/4.jpg'  ##  parameters ###
img = Image.open(image_file_path)   # [H W C]
img_width, img_height = img.size
# plt.imshow(img)
# plt.show()
img.save(r'data/original_'+ NAME_NET +'.jpg') # save original image

# image pre-processing
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(
    input_size=INPUT_SIZE, color_mean=color_mean, color_std=color_std)
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)   # [H W]
#p_palette = anno_class_img.getpalette()
## back,red,vege,yell,vehicle,bicy,build,pole,wall,person,green,traffic-sign,motorcy,road,sidewa,traffic-light,fence
# p_palette=[0,0,0,228,14,19,0,166,71,186,137,40,45,40,154,159,43,171,221,55,154,134,143,150,80,36,36,192,0,0,2,166,45,6,158,170,178,168,78,192,128,128,0,64,0,76,15,149,83,183,185]
p_palette=[0,0,0,228,14,19,0,166,71,186,137,40,45,40,154,159,43,171,221,55,154,134,143,150,80,36,36,192,0,0,255,255,255,6,158,170,178,168,78,192,128,128,0,64,0,76,15,149,83,183,185]
phase = "val" # the pre-processing method in test is the same as that in validation 
img, anno_class_img = transform(phase, img, anno_class_img)


# inference
net.eval()
x = img.unsqueeze(0).to(device)  # mini batch：torch.Size([1, 3, 475, 475])
outputs = net(x)

for key, value in outputs.items():
    y = value

# get the biggest class from PSPNet's output, change to pallet, change to original size
y = y[0].to('cpu').detach().numpy() # y: [2, 475, 475], float32

y = np.argmax(y, axis=0)  # y: [475, 475], int 0, 1

anno_class_img = Image.fromarray(np.uint8(y), mode="P")
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
anno_class_img.putpalette(p_palette)
# plt.imshow(anno_class_img)
# plt.show()
anno_class_img.save(r'data/mask_' + NAME_NET + '.png') # save mask image


# visualization
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
            # 150 : clarity
img = Image.open(image_file_path)   # [H W C]
result = Image.alpha_composite(img.convert('RGBA'), trans_img)
# plt.imshow(result)
# plt.show()
result.save(r'data/visualization_' + NAME_NET + '.png') # save visualization image
