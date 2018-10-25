import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import imageio
from utils.utils import get_args, process_config
from data_loader.data_loader import DataLoader
from models.cyclegan_attr_model import CycleGANAttrModel
from trainers.data_generator import DataGenerator

# TODO: Add proper logging

def main():
    # get json configuration filepath from the run argument
    # process the json configuration file
    args = get_args()
    config = process_config(args.config)

    print('Create the data generator')
    data_loader = DataLoader(config)

    print('Create the model')
    model = CycleGANAttrModel(config, is_train=False)
    predict_set = config['predict_set']  # either a, b, both
    model.build_predict_model(predict_set)
    print('model ready loading data now')

    os.makedirs('images/%s' % config['dataset_name'], exist_ok=True)

    if predict_set=='both' or predict_set=='a':
        testA_datagen = DataGenerator(img_filenames=data_loader.get_testA_data(), batch_size=1, target_size=(config['predict_img_height'], config['predict_img_width']))
        testA_generator = iter(testA_datagen)

        num_images = len(testA_datagen)
        for i in range(num_images):
            imgs_A = next(testA_generator)
            fake_B = model.predict_g_AB.predict(imgs_A)
            imageio.imwrite("images/%s/a_transl_%i.png" % (config['dataset_name'], i), ((fake_B[0]+1)*127.5).astype(np.uint8))

            if predict_set=='both':
                reconstr_A = model.predict_g_BA.predict(fake_B)
                imageio.imwrite("images/%s/a_recon_%i.png" % (config['dataset_name'], i), ((reconstr_A[0]+1)*127.5).astype(np.uint8))

    if predict_set=='both' or predict_set=='b':
        testB_datagen = DataGenerator(img_filenames=data_loader.get_testB_data(), batch_size=1, target_size=(config['predict_img_height'], config['predict_img_width']))
        testB_generator = iter(testB_datagen)

        num_images = len(testB_datagen)
        for i in range(num_images):
            imgs_B = next(testB_generator)    
            fake_A = model.predict_g_BA.predict(imgs_B)
            imageio.imwrite("images/%s/b_transl_%i.png" % (config['dataset_name'], i), ((fake_A[0]+1)*127.5).astype(np.uint8))

            if predict_set=='both':
                reconstr_B = model.predict_g_AB.predict(fake_A)
                imageio.imwrite("images/%s/b_recon_%i.png" % (config['dataset_name'], i), ((reconstr_B[0]+1)*127.5).astype(np.uint8))

if __name__ == '__main__':
    main()