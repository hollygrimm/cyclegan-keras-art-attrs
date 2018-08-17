# CycleGAN on Art Composition Attributes

## Requirements
Keras version 2.1.2

INSTALL keras-contrib for Instance Normalization:
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization.py

```
source activate tensorflow
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
```

Pre-trained ResNet50 Model with Art Composition Attributes: https://github.com/art-composition-cnn

## Download Dataset
$ bash download_dataset.sh apple2orange

## Run Training
```
python main.py -c input_params.json
```
