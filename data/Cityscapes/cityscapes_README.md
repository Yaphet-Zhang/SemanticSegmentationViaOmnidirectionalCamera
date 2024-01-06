# references
## official
- [cityscapes](https://www.cityscapes-dataset.com/)

- [cityscapes processing scripts](https://github.com/mcordts/cityscapesScripts)


## blog
- [语义分割-CityScapes数据集](https://blog.csdn.net/qq_42178122/article/details/116117762)

- [Cityscapes数据集的深度完整解析](https://blog.csdn.net/MVandCV/article/details/115331719)

- [将Cityscape转换为PASACAL VOC格式的目标检测数据集](https://blog.csdn.net/weixin_36670529/article/details/107301950)


# cityscapes (fine)
```
- label(30) 
    - flat: road, sidewalk, parking, rail track, 
    - human: person, rider
    - vehicle: car, truck, bus, on rails, motorcycle, bicycle, caravan, trailer
    - construction: building, wall, fence, guard rail, bridge, tunnel
    - object: pole, pole group, traffic sign, traffic light
    - nature: vegetation, terrain
    - sky: sky
    - void: ground, dynamic, static
- ignore label(5)
    - 0:'unlabeled', 1:'ego vehicle', 2: 'rectification border', 3:'out of roi', -1:'license plate'

- leftImg8bit(train): 5000
    - train: 2975, 18 citys
    - val: 500, 3 citys
    - test: 1525, 6 citys

- gtFine(label): 5000
    - train: 2975, 18 citys
    - val: 500, 3 citys
    - test: 1525, 6 citys

- image size: 1024*2048

- VOC image channel
    - img: [H, W, C] nint8 0-255
    - label: [H, W] nint8 0-5

- Cityscapes image channel 
    - img: [H, W, C] nint8 0-255
    - label: [H, W] nint8 0-5
    - color: [H, W, C] nint8 0-255, RGBA
```