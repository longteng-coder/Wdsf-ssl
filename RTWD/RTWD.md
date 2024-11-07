# RTWD

## Pretraining
The data that can be used for self-supervised pretraining are in path [RTWD/pre_train/](pre_train).

Defect images are in path [images/](pre_train/images). 
Labels are in path [labels/](pre_train/labels). 

Each txt provides an index of the defect type, and the coordinates of the 
defect circumscribed rectangle in the format [x0,x1,y0,y1].

The [label.txt](pre_train/label.txt) gives the correspondence between the type and the index.

The following table shows the number distribution of each type:

| Type name | Porosity | Slag | Lack of penetration | Lack of fusion | Crack | Undercut | Concave | Burn through |
|:----------|:--------:|-----:|:-------------------:|:--------------:|:-----:|:--------:|:-------:|:------------:|
| Quantity  |   3552   | 4833 |         42          |      567       |  16   |   577    |   340   |     515      |



## RTWD-Segmention
RTWD-Segmention is a data subset used to train and evaluate defect segmentation models.
It is located in path [RTWD/RTWD-Segmention/](RTWD-Segmention).

Defect images are in [RTWD-Segmention/Images/](RTWD-Segmention/Images).

The file name lists of the training set and the validation set are in [RTWD-Segmention/ImageSets/](RTWD-Segmention/ImageSets). The ratio of the two is approximately 1:1.
The training set data is also used for pre-training.

The ground truth segmentation masks are in [RTWD-Segmention/SegmentationClass/](RTWD-Segmention/SegmentationClass). 

The following table shows the number distribution of each type: 

| Type name | Porosity | Slag | Lack of penetration | Lack of fusion | Crack | Undercut | Concave | Burn through |
|:----------|:--------:|-----:|:-------------------:|:--------------:|:-----:|:--------:|:-------:|:------------:|
| Quantity  |   747    |  642 |         26          |      295       |  15   |   151    |   128   |     103      |
