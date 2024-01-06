## Goal
``` 
annotated(.json) >> VOCdataset
```

## Packages
```
pip install imgviz==0.12.1
pip install labelme==4.5.7
pip install opencv-python==3.4.9.31
```

## Usage
### write labels.txt file
```
__ignore__
_background_
pole
vegetation...
```
### write labels_mask.txt file
```
__ignore__
_background_
mask
```
### create VOCzbw
```
cd ./data/VOCdevkit
python labelme2voc.py annotated VOCzbw --labels labels.txt
```
### create VOCzbw_mask
```
cd ./data/VOCdevkit
python labelme2voc_mask.py annotated_mask VOCzbw_mask --labels labels_mask.txt
```
## Annotation
```
- annotate
    - obstacle(tree trunks & poles): adjacent
    - road, sidewalk: current, adjacent
    - crosswalk, yellow-warning-block: all
- mask
    - obstacle(tree trunks & poles): faraway, small 
    - road, sidewalk: faraway
    - crosswalk, yellow-warning-block: nothing
```
