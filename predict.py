import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import imageio
from utils.utils import get_args, process_config, create_dirs
from data_loader.data_loader import DataLoader
from models.cyclegan_attr_model import CycleGANAttrModel
from trainers.data_generator import DataGenerator

# TODO: Add proper logging

def main():
    # get json configuration filepath from the run argument
    # process the json configuration file
    try:
        args = get_args()
        # TODO: Error if args.config doesn't exist
        config, log_dir, checkpoint_dir = process_config(args.config)
    except:
        print('missing or invalid arguments')
        print('Unexpected error:', sys.exc_info()[0])

    # create the experiment directories
    create_dirs([log_dir, checkpoint_dir])

    print('Create the data generator')
    data_loader = DataLoader(config)

    print('Create the model')
    model = CycleGANAttrModel(config)
    model.build_predict_model()
    print('model ready loading data now')

    os.makedirs('images/%s' % config['dataset_name'], exist_ok=True)
    r, c = 2, 3

    testA_datagen = DataGenerator(img_filenames=data_loader.get_testA_data(), batch_size=1, target_size=(config['predict_img_size_x'], config['predict_img_size_y']))
    testB_datagen = DataGenerator(img_filenames=data_loader.get_testB_data(), batch_size=1, target_size=(config['predict_img_size_x'], config['predict_img_size_y']))

    testA_generator = iter(testA_datagen)
    testB_generator = iter(testB_datagen)

    imgs_A = next(testA_generator)
    imgs_B = next(testB_generator)

    # Translate images to the other domain
    fake_B = model.predict_g_AB.predict(imgs_A)
    fake_A = model.predict_g_BA.predict(imgs_B)

    # Translate back to original domain
    reconstr_A = model.predict_g_BA.predict(fake_B)
    reconstr_B = model.predict_g_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Plot images
    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/figure.png" % (config['dataset_name']))
    plt.close()

    imageio.imwrite("images/%s/a_transl.png" % (config['dataset_name']), ((fake_B[0]+1)*127.5).astype(np.uint8))
    imageio.imwrite("images/%s/b_transl.png" % (config['dataset_name']), ((fake_A[0]+1)*127.5).astype(np.uint8))
    imageio.imwrite("images/%s/a_recon.png" % (config['dataset_name']), ((reconstr_A[0]+1)*127.5).astype(np.uint8))
    imageio.imwrite("images/%s/b_recon.png" % (config['dataset_name']), ((reconstr_B[0]+1)*127.5).astype(np.uint8))

if __name__ == '__main__':
    main()