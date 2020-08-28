## Preprocessing SceneNet Dataset for training

### Requirements
- python3
- numpy
- Pillow
- os
- tqdm
- argparse
- scenenet_pb2.py
- NYU_WNID_TO_CLASS.json

The *scenenet_pb2.py* and *NYU_WNID_TO_CLASS.json* files is also required and must be included in the repository.
See the [SceneNet Repository] (https://github.com/jmccormac/pySceneNetRGBD) for more information.

### Usage



To use this script to preprocess data from SceneNet Dataset, 
- Download the dataset. The dataset is divided into 17 subfolders each having a number. It can be found in the [SceneNet Dataset] (https://robotvault.bitbucket.io/scenenet-rgbd.html)
- Download the protobuf file associated with the downloaded files. Each subfolder contains a set of images contained in several hundred folders. For example, the train_0 folder contains 999 image subfolders.
Ensure that each of this folders has a *photo* folder and *instance* folder. If not, you will need to modify the following lines (27-28) in *data_scenenet.py* to avoid errors.
```python
		path_images = os.path.join(path, 'photo')
		path_labels = os.path.join(path, 'instance')
```
- Run the file *data_scenenet.py* file


The script to run is used as follows:
```
Build the SceneNet dataset label

positional arguments:
  data_path             the path of the dataset
  protobuf_path         the path of the protobuf file
  train_rep             the path of the subfile where the data is. For example
                        if you are preprocessing rep train5 and the subfile
                        10, put 5/10

optional arguments:
  -h, --help            show this help message and exit
  --create-color CREATE_COLOR
                        create the color mask or not
```

For example to process the images from sub-folder 5 in the train_0 folder, do
```
$ python3 data_scenenet.py path/to/train0/to/folder5/ path/to/train0_protobuf/ 0/5
```

### Utilisation du script generation.sh

To preprocess all the folders from a training folder, you can use the **generation.sh** file.
To use it, make

``` shell
$ ./generation.sh path/to/train_X/ path/to/protobuf/ X
```

where :
 - path/to/train_X/ : the file to the training folder
 - path/to/protobuf/ : the path to the training protobuf file
 - X : the number of the train folder (0 for train_0)

 Ensure that the number of subfolders in the train folders is equal to 999. 
 Else you can modify the for clause in *generation.sh* in line 42.

 The *generation.sh* file can allow you to rename all the images and theirs labels in order to avoid conflicts when all the subfolders will be merged together.
 For that uncomment line 43 in *generation.sh* file and ensure that the file **rename.py** from *util* folder is in the current repository.
 Report to the README file of *util* folder fore further explanations.


