#!/bin/bash
git clone https://github.com/hollygrimm/cyclegan-keras-art-attrs.git
cd cyclegan-keras-art-attrs/
bash download_dataset.sh apple2orange

cd ~
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
source activate tensorflow_p36
pip install keras
pip install pillow
python setup.py install
source deactivate
cd ../cyclegan-keras-art-attrs/data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A1FvTA-n7EZrtLx7TD9q3KgF5khpAjVW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A1FvTA-n7EZrtLx7TD9q3KgF5khpAjVW" -O art_composition_cnn_weights.hdf5 && rm -rf /tmp/cookies.txt
if ! echo "d922aa82e6e67177915895e34f02e03e89a902d7a15914edcee0c3056f285d24 art_composition_cnn_weights.hdf5" | sha256sum -c -; then
    echo "Checksum failed" >&2
    exit 1
fi
