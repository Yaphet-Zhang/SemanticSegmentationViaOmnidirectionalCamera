import torchvision
import torch
from utils.hyperparameters import NUM_CLASSES


DeeplabV3 = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False, 
        progress=True, 
        num_classes=NUM_CLASSES, 
        aux_loss=None)


if __name__=='__main__':

    net = DeeplabV3
    print(net)
    print('====================')

    x = torch.randn(4, 3, 475, 475)
    print('x: {}'.format(x.shape))
    y = net(x)

    for key, value in y.items():
        y = value

    print('y: {}'.format(y.shape))
    print('====================')

    print('parameters numbers:', sum(param.numel() for param in net.parameters()))
    print('====================')



