import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from base.base_trainer import BaseTrain
from trainers.tensorboard_batch_monitor import TensorBoardBatchMonitor
from trainers.data_generator import DataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import imageio

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
                # FIXME: -{d_loss:.2f}
                filepath = os.path.join(self.checkpoint_dir, '%s-{epoch:02d}.hdf5' % self.config['exp_name']),
                monitor = self.config['checkpoint_monitor'],
                mode = self.config['checkpoint_mode'],
                save_best_only = self.config['checkpoint_save_best_only'],
                save_weights_only = self.config['checkpoint_save_weights_only'],
                verbose = self.config['checkpoint_verbose'],
            )
        )

        self.callbacks.append(
            TensorBoardBatchMonitor(
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

    def train(self):
        start_time = datetime.datetime.now()
        epoch = 0
        epochs = self.config['nb_epoch']
        batch_size = self.config['batch_size']

        valid = np.ones((batch_size,) + self.model.disc_patch)
        fake = np.zeros((batch_size,) + self.model.disc_patch)

        trainA_datagen = DataGenerator(img_filenames=self.trainA_data, batch_size=batch_size, target_size=(self.config['img_height'], self.config['img_width']))
        trainB_datagen = DataGenerator(img_filenames=self.trainB_data, batch_size=batch_size, target_size=(self.config['img_height'], self.config['img_width']))

        steps_per_epoch = len(trainA_datagen)
        print(steps_per_epoch)

        self.callbacks[0].set_model(self.model.combined)
        self.callbacks[1].set_model(self.model.combined)

        batch_logs = {}
        for epoch in range(epochs):
            print("start of epoch {}".format(epoch))
            steps_done = 0
            trainA_generator = iter(trainA_datagen)
            trainB_generator = iter(trainB_datagen)
            while steps_done < steps_per_epoch:
                imgs_A = next(trainA_generator)
                imgs_B = next(trainB_generator)

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

                outputs = [valid, valid, # y: ones
                           imgs_A, imgs_B, # compare g_BA(g_AB(img_A)) to y: imgs_A, g_AB(g_BA(img_B)) to y: imgs_B
                           imgs_A, imgs_B, # compare g_BA(img_A) to y: imgs_A, compare g_AB(img_B) to y: imgs_B
                          ]

                if self.config['add_perceptual_loss']:
                    percept_A = self.model.model_perceptual.predict(imgs_A)
                    percept_B = self.model.model_perceptual.predict(imgs_B)
                    percept_fake_A = self.model.model_perceptual.predict(fake_A)
                    percept_fake_B = self.model.model_perceptual.predict(fake_B)

                    outputs.extend([percept_fake_B, percept_fake_A,
                                   percept_fake_A, percept_fake_B,
                                   percept_A, percept_B])

                if self.config['numerical_target_attr_values']:
                    attr_outputs = []
                    
                    # used to normalize attribute value
                    min_val = 1
                    max_val = 10
                    for k, value in self.config['numerical_target_attr_values'].items():
                        # normalize
                        norm_value = (2 * ((value - min_val)/(max_val - min_val))) - 1
                        target_values = np.ones((batch_size, )) * norm_value
                        attr_outputs.extend([target_values, target_values])
                    outputs.extend(attr_outputs)

                if self.config['categorical_target_attr_values']:
                    self.color_encoder = LabelEncoder()
                    self.color_encoder.fit(self.config['colors'])

                    self.harmony_encoder = LabelEncoder()
                    self.harmony_encoder.fit(self.config['harmonies'])    
                    attr_outputs = []
                    for k, value in self.config['categorical_target_attr_values'].items():
                        if k == "pri_color":
                            color = self.color_encoder.transform([value])
                            value = to_categorical(color, num_classes=len(self.color_encoder.classes_))
                        elif k == "harmony":
                            harmony = self.harmony_encoder.transform([value])
                            value = to_categorical(harmony, num_classes=len(self.harmony_encoder.classes_))
                        target_values = np.ones((batch_size, )) * value
                        attr_outputs.extend([target_values, target_values])
                    outputs.extend(attr_outputs)

                # Combine discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # Train the generators
                g_loss = self.model.combined.train_on_batch([imgs_A, imgs_B],
                                                        outputs)

                batch_logs['D Loss'] = d_loss[0]
                batch_logs['D Acc'] = 100*d_loss[1]
                batch_logs['G Loss'] = g_loss[0]
                batch_logs['G Adversarial Loss'] = np.mean(g_loss[1:3])
                batch_logs['G Cycle-Consistency Loss'] = np.mean(g_loss[3:5]) # if we translate from one domain to another, and then back again
                batch_logs['G Identity Loss'] = np.mean(g_loss[5:6]) # real samples of the target domain (e.g. img_A) is provided as input to the generator (B->A)    
                loss_index = 7

                if self.config['add_perceptual_loss']:
                    batch_logs['G Feature Loss Fake B'] = g_loss[loss_index]
                    batch_logs['G Feature Loss Fake A'] = g_loss[loss_index + 1]
                    batch_logs['G Feature Loss Fake A Recon B'] = g_loss[loss_index + 2]
                    batch_logs['G Feature Loss Fake B Recon A'] = g_loss[loss_index + 3]
                    batch_logs['G Feature Loss Real B Recon A'] = g_loss[loss_index + 4]
                    batch_logs['G Feature Loss Real B Recon B'] = g_loss[loss_index + 5]
                    loss_index += 6

                if self.config['numerical_target_attr_values']:
                    for i, k in enumerate(self.config['numerical_target_attr_values'].keys()):
                        batch_logs['G {} Loss Fake A'.format(k)] = g_loss[(i * 2) + loss_index]
                        batch_logs['G {} Loss Fake B'.format(k)] = g_loss[(i * 2) + loss_index + 1]
                    loss_index += len(self.config['numerical_target_attr_values']) * 2

                if self.config['categorical_target_attr_values']:
                    for i, k in enumerate(self.config['categorical_target_attr_values'].keys()):
                        batch_logs['G {} Loss Fake A'.format(k)] = g_loss[(i * 2) + loss_index]
                        batch_logs['G {} Loss Fake B'.format(k)] = g_loss[(i * 2) + loss_index + 1]                   

                self.callbacks[1].on_batch_end((epoch * steps_per_epoch) + steps_done, batch_logs)

                steps_done += 1

                # If at save interval => save generated image samples
                if steps_done % self.config['sample_interval'] == 0:
                    self.sample_images(epoch, steps_done)
            epoch += 1

            # save checkpoint
            self.callbacks[0].on_epoch_end(epoch)
            
            # shuffle indices
            trainA_datagen.on_epoch_end()
            trainB_datagen.on_epoch_end()
                    

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.config['dataset_name'], exist_ok=True)
        r, c = 2, 3

        testA_datagen = DataGenerator(img_filenames=self.testA_data, batch_size=1, target_size=(self.config['img_height'], self.config['img_width']))
        testB_datagen = DataGenerator(img_filenames=self.testB_data, batch_size=1, target_size=(self.config['img_height'], self.config['img_width']))

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

        imageio.imwrite("images/%s/%d_%d_a_transl.png" % (self.config['dataset_name'], epoch, batch_i), ((fake_B[0]+1)*127.5).astype(np.uint8))
        imageio.imwrite("images/%s/%d_%d_b_transl.png" % (self.config['dataset_name'], epoch, batch_i), ((fake_A[0]+1)*127.5).astype(np.uint8))
        imageio.imwrite("images/%s/%d_%d_a_recon.png" % (self.config['dataset_name'], epoch, batch_i), ((reconstr_A[0]+1)*127.5).astype(np.uint8))
        imageio.imwrite("images/%s/%d_%d_b_recon.png" % (self.config['dataset_name'], epoch, batch_i), ((reconstr_B[0]+1)*127.5).astype(np.uint8))

        