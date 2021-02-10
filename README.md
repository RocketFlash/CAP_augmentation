# "Cut and paste" augmentation

Repository contains easy to use Python implementation of "Cut and paste" augmentation for object detection and instance segmentation.

## Installation

To install all requirements run:

```bash
pip install -r requirements.txt
```

### Requirements

  * Python 3
  * OpenCV
  * numpy


## Generate pedestrians dataset from CityScapes and CityPersons

Put Cityscapes and CityPersons datasets in data folder. Edit parameters in dataset/config.py if you want and then just run:

```bash
./dataset/generate_and_filter_dataset.sh 
```

This script will create a dataset of png images cutted and filtered in the data/human_dataset_filtered folder or in the folder that you specified in the data/config.py file.

Another option is to run python scripts manually step by step. First, we need to create .png files of people using instance masks from cityscapes dataset:

```bash
python dataset/generate_dataset.py 
```

Next, we need to filter images to remove too small or too cropped (only a small part of the body is visible) images:

```bash
python dataset/filter_dataset.py 
```

## Example of usage

```python
from src.cap_aug import CAP_AUG
import cv2

SOURCE_IMAGES = ['list/', 'of/', 'paths/', 'to/', 'the/', 'source/', 'image/', 'files']

image = cv2.imread('path/to/the/destination/image')

cap_aug = CAP_AUG(SOURCE_IMAGES, n_objects_range=[10,20],        
                                        h_range=[100,101],
                                        x_range=[500, 1500],
                                        y_range=[600 ,1000],
                                        coords_format='xyxy')
result_image, result_coords = cap_aug(image)
```