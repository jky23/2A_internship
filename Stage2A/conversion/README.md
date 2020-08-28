# Conversion from Caffe to TensorFlow/Keras

This directory is an attempt to convert a model from the Caffe platform to the Keras platform.

These scripts will take your Caffe model and first convert the weight stored in your *caffemodel* file and then re-create the architecture by following your *prototxt* file.

The *weight converter* is based on the code of [Caffe-to-Keras Weight Converter](https://github.com/pierluigiferrari/caffe_weight_converter) repository.

The layers currently implemented are :

- Convolutional Layer
- BatchNorm Layer
- Interp Layer
- ReLU Layer
- MemoryData Layer
- Pooling Layer
- Scale Layer
- Concat Layer
- Dropout Layer
- SoftMax layer


## Installation

To use this code you need to install Caffe (**PyCaffe**) according to Caffe Installation instructions from [Caffe PSPNet](https://github.com/hszhao/PSPNet).
Do not install Caffe from the Official repository, else you will not be able to load properly your *Caffe model*.

The requirements can be install with

```shell
pip install -r requirements.txt
```

## Usage

To use this code make,

```shell
python convert_to_keras.py -h

usage: convert_to_keras.py [-h] [-v] prototxt caffemodel model_name

Convert a PSPNet model from the Caffe format to Tensorflow/Keras format

positional arguments:
  prototxt       the path to the prototxt file
  caffemodel     the path to the caffemodel file
  model_name     the name of the model. It will be use to create the file
                 where the model will be save

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  Print the information while conversion

```

For example, if you make

```shell
python convert_to_keras.py your_protofile.protoxt your_caffe_weights.caffemodel my_model
```

it will output a **my_model.h5** file which is your model in Keras Format.

## Resources

- [Caffe PSPNet](https://github.com/hszhao/PSPNet)
- [Caffe-to-Keras Weight Converter](https://github.com/pierluigiferrari/caffe_weight_converter)