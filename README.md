# CycleGAN on Art Composition Attributes

Please read the accompanying blog post: [https://hollygrimm.com/acan_final](https://hollygrimm.com/acan_final)

## Requirements
* Keras version 2.1.2
* keras-contrib from Aug 9, 2018 hash: 3427000d9fa21561c31c01479fa74fba1a36ab08
* pillow
* Weights from https://github.com/hollygrimm/art-composition-cnn

## AWS Install
* Select Deep Learning AMI (Ubuntu) Version 13.0
* Instance Type `GPU Compute` such as p2.xlarge
* 125GB sda1

Connect to instance, copy contents of [aws-setup.sh](aws-setup.sh) to file in /home/ubuntu and run:
```
vi aws-setup.sh
chmod +x aws-setup.sh
./aws-setup.sh
```

## Manual Install
### keras-contrib install
```
source activate tensorflow
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
git reset --hard 3427000d9fa21561c31c01479fa74fba1a36ab08
python setup.py install
```

### Download Art Composition Attributes Network Weights
Weights can be downloaded with these commands:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A1FvTA-n7EZrtLx7TD9q3KgF5khpAjVW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A1FvTA-n7EZrtLx7TD9q3KgF5khpAjVW" -O art_composition_cnn_weights.hdf5 && rm -rf /tmp/cookies.txt

sha256sum d922aa82e6e67177915895e34f02e03e89a902d7a15914edcee0c3056f285d24
```

Or train your own weights using this repository: https://github.com/hollygrimm/art-composition-cnn

### Download Dataset
```
bash download_dataset.sh apple2orange
```

## Run Training
```
source activate tensorflow_p36
cd cyclegan-keras-art-attrs/
python main.py -c input_params.json
```

## Run Tests
```
cd tests
python cyclegan_keras_art_attrs_tests.py
```


## Acknowledgements

* Jun-Yan Zhu https://github.com/junyanz/CycleGAN
* Erik Linder-Norén https://github.com/eriklindernoren/Keras-GAN
* HagopB https://github.com/HagopB/cyclegan
* Ulyanov et al Instance Normalization: The Missing Ingredient for Fast Stylization https://arxiv.org/pdf/1607.08022.pdf
* Ahmed Hamada Mohamed Kamel El-Hinidy https://github.com/Ahmkel/Keras-Project-Template



