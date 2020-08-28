## Preprocessing ADE20K Dataset for training

### Requirements
- python3
- numpy
- pandas
- Pillow
- os
- tqdm
- argparse

The *objectInfo150.csv* file is also required. It list all the classes that can be found in the datasets.

### Usage

To use this script to preprocess data from ADE20K Dataset, 
- Download the dataset.
  The dataset can be download [ADE20K Datasets] (http://sceneparsing.csail.mit.edu/)
 Ensure that the images are in a *images/* folder and the label are
 in a *annotations/* folder. Otherwise, you can modify the following lines (20-21) in the *data_ade.py* file :
```python
        image_path = os.path.join(path, 'images')
        label_path = os.path.join(path, 'annotations')
```
- Run the file *data_nyu_sunrgb_uiuc.py* file

The script to run is used as follows:
```
Build the ADE20K dataset label

positional arguments:
  data_path             the path of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --create-color CREATE_COLOR
                        create the color mask or not
```

For example to process the images, do
```
$ python3 data_ade.py path/to/dataset/
```



