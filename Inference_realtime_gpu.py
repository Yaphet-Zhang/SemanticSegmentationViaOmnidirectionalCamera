from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from model.pspnet import PSPNet

##########  parameters  ##########
n_classes=13
##########  parameters  ##########

# load network
net = PSPNet(n_classes=n_classes)
state_dict = torch.load("./weights/pspnet_110.pth",  #####  parameters  #####
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)
print("##### Successful loaded your model #####")

########## use 1-GPU ##########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("##### Your device is", device,"#####")
net=net.to(device)
########## use 1-GPU ##########

########## use Multi-GPU ##########
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count()> 1:
#    net=nn.DataParallel(net)
#print("##### Your device is", device,"#####")
#net.to(device)
########## use Multi-GPU ##########

from utils.dataloader import make_datapath_list, DataTransform

# image pre-processing
rootpath = "./data/VOCdevkit/VOCzbw/"  #####  parameters  #####
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    rootpath=rootpath)
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)  # [H W]
p_palette = anno_class_img.getpalette()
# anno_class_img 
anno_class_img = np.array(anno_class_img)  # [H W]
index = np.where(anno_class_img == 255)
anno_class_img[index] = 0
anno_class_img = torch.from_numpy(anno_class_img)

##########  parameters  ##########
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
input_size=475
camera_id=0
PIL_alpha=180
cv2_alpha=0.9
cv2_beta=1
cv2_gamma=15
#########  parameters  ##########

cap=cv2.VideoCapture(camera_id)
if cap.isOpened() :
    print("##### Successfully open your camera #####")
else:
    print("##### Failed to open your camera #####")

while cap.isOpened():
    ret,frame=cap.read()
    # [H W C] 
    img_height=frame.shape[0]
    img_width=frame.shape[1]
    
    ##########  parameters  ##########
    # resolution of video  
    img_height_resolution=int(img_height/4)
    img_width_resolution=int(img_width/4)
    # window size of video
    img_height_show=int(img_height*1)
    img_width_show=int(img_width*1)
    ##########  parameters  ##########

    frame=cv2.resize(frame,(input_size,input_size),interpolation=cv2.INTER_CUBIC) # [W H]        
    # OpenCV(ndarray) to Tensor  &  (RGB)/255 for (0-1)  &  color_mean + color_standar_deviation
    transformer=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(color_mean,color_std)
    ])
    frame_tensor=transformer(frame) #[C H W]

    # start evaluation mode
    net.eval()
    # torch(img): [3,475,475] -->> torch of net_input(x): ([1,3,475,475]) // [C H W] --> [N C H W] 
    x = frame_tensor.unsqueeze(0)    
    # send the input data to device(GPU or CPU)
    x=x.to(device)
    outputs = net(x)
    y = outputs[0]  # torch of net_output(y): [1,10,475,475] // [N C H W]
    y=y[0]  # torch(y): [1,10,475,475] -->> [10,475,475] // [N C H W] --> [C H W]
    # get the biggest class from PSPNet's output by 'argmax'
    y=torch.argmax(y,dim=0)
    # change to CPU for using pillow  
    y=y.cpu().numpy()
    anno_class_img = Image.fromarray(np.uint8(y))  # [475,475]
    anno_class_img = anno_class_img.resize((img_width_resolution, img_height_resolution), Image.NEAREST) # [W H]    
    # put palette
    anno_class_img.putpalette(p_palette) # [W H]    
   
    # draw a black background & color it 
    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')            
    for x in range(img_width_resolution):
        for y in range(img_height_resolution):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            else:
                trans_img.putpixel((x, y), (r, g, b, PIL_alpha))
    
    ##### use PIL to composite #####
    # read original image & resize
#     img = Image.fromarray(np.uint8(frame))
#     img=img.resize((img_width_resolution,img_height_resolution),Image.NEAREST) # [W H]    
    # composite original image & mask image
#     result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    # PIL(RGB) to ndarray to OpenCV(BGR)
#     result_frame=cv2.cvtColor(np.asarray(result),cv2.COLOR_RGB2BGR)
    ##### (Right orther: RGB) #####
    
    ##### use cv2 to composite #####
    # read original image & resize
    img=cv2.resize(frame,(img_width_resolution,img_height_resolution),interpolation=cv2.INTER_CUBIC) # [W H]
    # PIL(RGB) to ndarray to OpenCV(BGR)
    trans_img=cv2.cvtColor(np.asarray(trans_img),cv2.COLOR_RGB2BGR)
    # composite original image & mask image
    result_frame=cv2.addWeighted(img,cv2_alpha,trans_img,cv2_beta,cv2_gamma)
    ##### (inverse orther: BGR) #####
    
    # create a window
    cv2.namedWindow('Images',0)
    # resize the window
    cv2.resizeWindow('Images',img_width_show,img_height_show) # [W H]
    # show your Demo
    cv2.imshow('Images',result_frame)           
    
    c=cv2.waitKey(1)
    if c==27:
        break
cap.release()
cv2.destroyAllWindows()  

# #### 速度比较：INTER_NEAREST（最近邻插值）>INTER_CUBIC(三次样条插值)>INTER_LINEAR(线性插值)>INTER_AREA  (区域插值)
# #### 对图像进行缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法。
# #### OpenCV推荐：如果要缩小图像，通常推荐使用#INTER_AREA插值效果最好，而要放大图像，通常使用INTER_CUBIC(速度较慢，但效果最好)，或者使用INTER_LINEAR(速度较快，效果还可以)。至于最近邻插值INTER_NEAREST，一般不推荐使用
# #### 在测试的时候，发现使用INTER_CUBIC方法，并不慢啊啊啊，比INTER_LINEAR还快！！！！这个就比较尴尬了！我猜是OpenCV有对INTER_CUBIC插值方法进行特殊的优化吧！
# 
