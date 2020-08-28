# Training PSPNet for Semantic Segmentation

## Contents

1. [Train locally](#1)
2. [Train on AWS Sagemaker](#2)
3. [Train on Paperspace](#3)

## 1. Train locally

### 1-1. Requirements

The scripts were written in Python and use the latest version of Pytorch.
You will need mainly:

- python3 (3.6+)
- torch (1.5.0+)
- numpy
- opencv
- Pillow
- pytorch-lightning
- tensorboard
- matplotlib

Theses packages can be installed with the *requirements.txt* file with

```shell
pip install -r requirements.txt
```

The architecture of the model is given by the files *model/resnet.py* and *model/pspnet.py* and are those from [PSPNet Segmentation Model](<https://github.com/hszhao/semseg>).
The model is based on a pretrained Resnet-50 model that is different from the one provided by PyTorch. You need to download this pretrained model and put in the **initmodel** file. This is the one called by default by the *resnet.py* file.
You can download it [here](<https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v>).

The parameters for training or test the model are in the file *params_pspnet50.yaml*.
You can report to this file to see the hyperparameters that will be set as your training/testing configurations.

### 1-2. Usage

1. Create the different folders for the datasets and the results. 

```shell
mkdir data/
mkdir data/train/
mkdir data/train/image/
mkdir data/train/mask/
mkdir data/val/
mkdir data/val/image/
mkdir data/val/mask/
mkdir data/test/
mkdir data/result/
```

 2. Download your datasets (train, val, test) and their masks(labels) and put them respectively in their folders.
Your datasets need to be already prepocessed (images and mask with the numbers of classes you need.)

 3. You may need to change some hyperparameters.
Make this command to see the differents hyperparameters you could tune.

```shell
python train.py --help
```

 4. You can launch training with

```shell
python train.py --epochs 100 --train_dir path/to/train/data/ --test_dir path/to/val/data/ --model_dir path/to/save/folder/
```

For training it is highly recommanded ([PSPNet Segmentation Model](<https://github.com/hszhao/semseg>)) to have a GPU instance (at least 8Gb GPU) and train for 100 epochs, to have significant results.
This code does not support CPU instance.

 5. You can also test the model on your own datasets.


For this purpose, run

```shell
python test.py
```

You do not need to specify them unlike when you run *model/train.py*. The hyperparameters will be taken directly from *config/params_pspnet50.yaml*.
You can change some to fit to your configurations.

For example,

- To test your model only change the ```has_prediction``` value in the *config/params_pspnet50.yaml* to ```False```.
- To evaluate your model on images and their labels to see the score ```has_prediction``` value in the *config/params_pspnet50.yaml* to ```True```.
  You can choose a metric between **Accuracy** and **mIoU (Intersection over Union)** by changing the ```metric``` value.
  You can also specify a threshold to only save images with poor or great scores.

For that change in the*config/params_pspnet50.yaml* file :

  1. *threshold* (a float number between 0 and 1)
  2. *condition* (a string between inf or sup)
  3. *model_path* (a string that specify the path to the model to use)
  4. *save_folder*  (a string that specify the path to the folder to save your images.)

Report to the *config/params_pspnet50.yaml* file for further information.

## 2. Train with AWS Sagemaker

Refer to the folder **sagemaker training** and read the *README.md* and the notebook file.

## 3. Train on Paperspace

You can also use [Paperspace](https://www.paperspace.com/) to train the model if you don't have enough ressources.
For that upload to your Paperspace account and create a Notebook.
A *GPU+ Machine* is sufficient for training but a *P5000 Machine* will speed up your training.

In the notebook upload in the **storage/** folder this repository. (in order to make your data persistent and upload automatically when you sign in and start your notebook)
You can create a **data/** folder inside and follow the architecture described in [Usage](#1.2).

You can train or make tests as you do when you train locally.
