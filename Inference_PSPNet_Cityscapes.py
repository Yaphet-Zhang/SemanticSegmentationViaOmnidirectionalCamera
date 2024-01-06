from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.hyperparameters import NUM_CLASSES, EPOCH, INPUT_SIZE, NAME_NET
from model.pspnet import PSPNet
import torch.utils.data as data




########## network ##########
net = PSPNet(n_classes=NUM_CLASSES)
net = nn.DataParallel(net) # 1-node, multi-GPU




########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
print('device: ', device)
print('====================')




########## load weights ##########
weights_path = 'weights/' + NAME_NET + '_' + str(EPOCH) + '_Cityscapes' + '.pth'
flag = net.load_state_dict(torch.load(weights_path , map_location=device))
print(flag)
print(weights_path + ': loaded!')
print('====================')





VOC_validatiopn = False # True or False

i = 0 # VOC validation data number
Cityscapes_specified = True # True or False




########## img & label path ##########
if VOC_validatiopn:
    from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
    rootpath = './data/VOCdevkit/VOCzbw/'
elif Cityscapes_specified:
    from utils.dataloader_Cityscapes import make_datapath_list, DataTransform, VOCDataset
    rootpath = './data/Cityscapes/VOCzbw/'
train_img_list, train_anno_list, val_img_list, val_anno_list, train_mask_list = make_datapath_list(rootpath=rootpath)




########## preparation ##########
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=INPUT_SIZE, color_mean=color_mean, color_std=color_std)
val_dataset = VOCDataset(val_img_list, val_anno_list, False, phase = 'val', transform=transform)




########## original ##########
if VOC_validatiopn:
    # image_file_path = val_img_list[i] ##### via validation data
    # result 1
    image_file_path = 'data/VOCdevkit/VOCzbw/JPEGImages/0006.jpg' ##### via specified path
    # result 2
    # image_file_path = 'data/VOCdevkit/VOCzbw/JPEGImages/0072.jpg' ##### via specified path
    # result 3
    # image_file_path = 'data/VOCdevkit/VOCzbw/JPEGImages/0039.jpg' ##### via specified path
elif Cityscapes_specified:
    image_file_path = 'data/Cityscapes/VOCzbw/JPEGImages/frankfurt_000000_000294_leftImg8bit.png'

print('img path: ', image_file_path)

img = Image.open(image_file_path)
img_width, img_height = img.size
img.save(r'data/Cityscapes_org_'+ NAME_NET +'.png')




########## inference ##########
if VOC_validatiopn:
    # anno_file_path = val_anno_list[i] ##### via validation data
    # result 1
    anno_file_path = 'data/VOCdevkit/VOCzbw/SegmentationClassPNG/0006.png' ##### via specified path
    # result 2
    # anno_file_path = 'data/VOCdevkit/VOCzbw/SegmentationClassPNG/0072.png' ##### via specified path
    # result 3
    # anno_file_path = 'data/VOCdevkit/VOCzbw/SegmentationClassPNG/0039.png' ##### via specified path
elif Cityscapes_specified:
    anno_file_path = 'data/Cityscapes/VOCzbw/SegmentationClassPNG/frankfurt_000000_000294_gtFine_labelIds.png'


print('label path: ', anno_file_path)
print('====================')

anno_class_img = Image.open(anno_file_path)
# pre-process
img, anno_class_img, mask_0_or_1 = transform('val', img, anno_class_img, False) 
# label
label = anno_class_img.long().to(device)
##### inference
net.eval()
x = img.unsqueeze(0).to(device)
output = net(x)
y = output[0]
# prediction
predict = torch.argmax(y, dim=1)[0] 




########## evaluation ##########
##### VOC
if VOC_validatiopn: 
    sum_obstacle, sum_road, sum_sidewalk = sum(sum(label==1)), sum(sum(label==2)), sum(sum(label==3))
    print('sum obstacle:', sum_obstacle.item(), 'sum road:', sum_road.item(), 'sum sidewalk:', sum_sidewalk.item())

    correct_obstacle = 0
    correct_road = 0
    correct_sidewalk = 0
    for i in range(475):
        for j in range(475):
            if predict[i][j] == 17 and label[i][j] == 1:
                correct_obstacle += 1
            elif predict[i][j] == 7 and label[i][j] == 2:
                correct_road += 1
            elif predict[i][j] == 8 and label[i][j] == 3:
                correct_sidewalk += 1

    if sum_obstacle != 0:
        print('obstacle acc:', (correct_obstacle/sum_obstacle).item())
    if sum_road != 0:
        print('road acc:', (correct_road/sum_road).item())
    if sum_sidewalk != 0:
        print('sidewalk acc:', (correct_sidewalk/sum_sidewalk).item())

    accuracy = torch.sum(predict == label).item() / (label.shape[0]*label.shape[1])
    print('accucay:', accuracy)


##### Cityscapes
elif Cityscapes_specified: 
    sum_obstacle, sum_road, sum_sidewalk = sum(sum(label==17)), sum(sum(label==7)), sum(sum(label==8))
    print('sum obstacle:', sum_obstacle.item(), 'sum road:', sum_road.item(), 'sum sidewalk:', sum_sidewalk.item())

    correct_obstacle = 0
    correct_road = 0
    correct_sidewalk = 0
    for i in range(475):
        for j in range(475):
            if predict[i][j] == 17 and label[i][j] == 17:
                correct_obstacle += 1
            elif predict[i][j] == 7 and label[i][j] == 7:
                correct_road += 1
            elif predict[i][j] == 8 and label[i][j] == 8:
                correct_sidewalk += 1

    if sum_obstacle != 0:
        print('obstacle acc:', (correct_obstacle/sum_obstacle).item())
    if sum_road != 0:
        print('road acc:', (correct_road/sum_road).item())
    if sum_sidewalk != 0:
        print('sidewalk acc:', (correct_sidewalk/sum_sidewalk).item())

    accuracy = torch.sum(predict == label).item() / (label.shape[0]*label.shape[1])
    print('accucay:', accuracy)




########## visualization ##########
##### draw mask
y = y[0].to('cpu').detach().numpy()
y = np.argmax(y, axis=0)
anno_class_img = Image.fromarray(np.uint8(y), mode='P')
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
# p_palette = anno_class_img.getpalette()
p_palette = [0, 0,  0,
             0,  0,  0,
            0,  0,  0,
            0,  0,  0,
            0,  0,  0,
            111, 74,  0,
            81,  0, 81,
            0,166,71,
            186,137,40,
            250,170,160,
            230,150,140,
            70, 70, 70,
            102,102,156,
            190,153,153,
            180,165,180,
            150,100,100,
            150,120, 90,
            228,14,19,
            153,153,153,
            250,170, 30,
            220,220,  0,
            107,142, 35,
            152,251,152,
            70,130,180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 0, 90,
            0, 0, 110,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
]

# p_palette = [0, 0,  0,
#              0,  0,  0,
#             0,  0,  0,
#             0,  0,  0,
#             0,  0,  0,
#             111, 74,  0,
#             81,  0, 81,
#             128, 64,128,
#             244, 35,232,
#             250,170,160,
#             230,150,140,
#             70, 70, 70,
#             102,102,156,
#             190,153,153,
#             180,165,180,
#             150,100,100,
#             150,120, 90,
#             153,153,153,
#             153,153,153,
#             250,170, 30,
#             220,220,  0,
#             107,142, 35,
#             152,251,152,
#             70,130,180,
#             220, 20, 60,
#             255, 0, 0,
#             0, 0, 142,
#             0, 0, 70,
#             0, 60, 100,
#             0, 0, 90,
#             0, 0, 110,
#             0, 80, 100,
#             0, 0, 230,
#             119, 11, 32,
# ]
anno_class_img.putpalette(p_palette)
anno_class_img.save(r'data/Cityscapes_mask_' + NAME_NET + '.png')




# ##### draw vis
# trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
# anno_class_img = anno_class_img.convert('RGBA')  
# for x in range(img_width):
#     for y in range(img_height):
#         pixel = anno_class_img.getpixel((x, y))
#         r, g, b, a = pixel
#         if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
#             continue
#         else:
#             trans_img.putpixel((x, y), (r, g, b, 80))
#             # 150 : clarity
# img = Image.open(image_file_path)   # [H W C]
# result = Image.alpha_composite(img.convert('RGBA'), trans_img)
# result.save(r'data/Cityscapes_vis_' + NAME_NET + '.png')
