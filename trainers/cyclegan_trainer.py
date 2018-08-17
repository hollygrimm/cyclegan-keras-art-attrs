import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard
from base.base_trainer import BaseTrain
from trainers.data_generator import DataGenerator

class CycleGANModelTrainer(BaseTrain):
    def __init__(self, model, trainA_data, trainB_data, testA_data, testB_data, config, tensorboard_log_dir, checkpoint_dir):
        super(CycleGANModelTrainer, self).__init__(model, trainA_data, trainB_data, testA_data, testB_data, config)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir   
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath = os.path.join(self.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config['exp_name']),
                monitor = self.config['checkpoint_monitor'],
                mode = self.config['checkpoint_mode'],
                save_best_only = self.config['checkpoint_save_best_only'],
                save_weights_only = self.config['checkpoint_save_weights_only'],
                verbose = self.config['checkpoint_verbose'],
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir = self.tensorboard_log_dir,
                write_graph = self.config['tensorboard_write_graph'],
                histogram_freq = 0, # don't compute histograms
                write_images = False # don't write model weights to visualize as image in TensorBoard
            )
        )

        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config['comet_api_key'], project_name=self.config['exp_name'])
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())      

    def train(self, sample_interval=200):
        # TODO: Add callbacks

        start_time = datetime.datetime.now()
        epoch = 0
        epochs = self.config['nb_epoch']
        batch_size = self.config['batch_size']

        valid = np.ones((batch_size,) + self.model.disc_patch)
        fake = np.zeros((batch_size,) + self.model.disc_patch)

        trainA_datagen = DataGenerator(img_filenames=self.trainA_data, batch_size=batch_size, target_size=(self.config['img_size'], self.config['img_size']))
        trainB_datagen = DataGenerator(img_filenames=self.trainB_data, batch_size=batch_size, target_size=(self.config['img_size'], self.config['img_size']))

        steps_per_epoch = len(trainA_datagen)
        print(steps_per_epoch)

        for epoch in range(epochs):
            steps_done = 0
            trainA_generator = iter(trainA_datagen)
            trainB_generator = iter(trainB_datagen)
            while steps_done < steps_per_epoch:
                imgs_A = next(trainA_generator)
                imgs_B= next(trainB_generator)

                # Translate images to opposite domain
                fake_B = self.model.g_AB.predict(imgs_A)
                fake_A = self.model.g_BA.predict(imgs_B)

                # Train the discriminators and calculate losses for A and B
                dA_loss_real = self.model.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.model.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.model.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.model.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Combine discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # Train the generators
                g_loss = self.model.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            steps_done, steps_per_epoch,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                steps_done += 1
                print(steps_done)

                # If at save interval => save generated image samples
                if steps_done % sample_interval == 0:
                    self.sample_images(epoch, steps_done)
            epoch += 1
            
            # shuffle indices
            trainA_datagen.on_epoch_end()
            trainB_datagen.on_epoch_end()
                    

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.config['dataset_name'], exist_ok=True)
        r, c = 2, 3

        testA_datagen = DataGenerator(img_filenames=self.testA_data, batch_size=1, target_size=(self.config['img_size'], self.config['img_size']))
        testB_datagen = DataGenerator(img_filenames=self.testB_data, batch_size=1, target_size=(self.config['img_size'], self.config['img_size']))

        testA_generator = iter(testA_datagen)
        testB_generator = iter(testB_datagen)

        imgs_A = next(testA_generator)
        imgs_B = next(testB_generator)

        # Translate images to the other domain
        fake_B = self.model.g_AB.predict(imgs_A)
        fake_A = self.model.g_BA.predict(imgs_B)

        # Translate back to original domain
        reconstr_A = self.model.g_BA.predict(fake_B)
        reconstr_B = self.model.g_AB.predict(fake_A)

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
        fig.savefig("images/%s/%d_%d.png" % (self.config['dataset_name'], epoch, batch_i))
        plt.close()
