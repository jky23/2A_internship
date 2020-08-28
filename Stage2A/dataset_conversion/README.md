### Preprocessing Dataset for Training

This directory contains the scripts to preprocess the images as well as their segmentation labels in order to conform to the different metalabels, before performing the training.
The supported datasets are:
- SceneNet Dataset
- ADE20K Dataset
- SunRGBD-NYU-UIUC Dataset.

All datasets must inherit from the **DataSetGeneric** class in the *data_generic.py* file.
Before preprocessing a dataset, you must create a folder for this dataset in the datasets implementation folder, then copy the following files to the created folder :
- *data_generic.py*
- *dict_labels.json*
- *meta-classes.json*

The *util* folder contains functions to retrieve images from a website, or rename images from a folder (see README in *util*)