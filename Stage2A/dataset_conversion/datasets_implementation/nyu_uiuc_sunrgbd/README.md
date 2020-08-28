## Preprocessing NYU-SunRGBD-UIUC Dataset for training

### Requirements
- python3
- numpy
- Pillow
- os
- tqdm
- argparse

The *names.txt* file is also required. It list all the classes that can be found in the datasets.

### Usage

To use this script to preprocess data from Nyu-SunRGBD-UIUC Dataset, 
- Download the dataset. 
  The dataset can be donwload [here] (http://82.255.28.28:8082/datasets/).
  Ensure that the images are in a *image/* folder and the label are
 in a *mask/* folder. Otherwise, you can modify the following lines (20-21) in the *data_nyu_sunrgb_uiuc.py* file :
```python
        image_path = os.path.join(path, 'image')
        label_path = os.path.join(path, 'mask')
```
- Run the file *data_nyu_sunrgb_uiuc.py* file

The script to run is used as follows:
```
Build the Nyu-SunRgbd-Uiuc dataset label

positional arguments:
  data_path             the path of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --create-color CREATE_COLOR
                        create the color mask or not
```

For example to process the images, do
```
$ python3 data_nyu_sunrgb_uiuc.py path/to/dataset/
```



