
from base.base_model import BaseModel
import keras.backend as K
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import Lambda, GlobalAveragePooling2D
from keras.optimizers import Adagrad

class CycleGANAttrModel(BaseModel):
    def __init__(self, config):
        super(CycleGANAttrModel, self).__init__(config)
        self.channels = 3
        self.img_size = config['img_size']
        self.img_shape = (self.img_size, self.img_size, self.channels)
        self.weights_path = config['weights_path']
        self.base_lr = config['base_lr']
        self.beta_1 = config['beta_1']
        self.loss_weights = config['loss_weights']
        self.target_attr_values = config['target_attr_values']
        self.comp_attrs_weights_path = config['comp_attrs_weights_path']
        self.add_perceptual_loss = config['add_perceptual_loss']
        self.build_model()

    def build_generator(self):

        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        d0 = Input(shape=self.img_shape)

        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def build_perceptual_model(self, input_shape, trainable=False, pop=True):
        # import ResNet50 pretrained on imagenet
        model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        if pop == True:
            model.layers.pop() # pop pooling layer
            model.layers.pop() # pop last activation layer

        for layer in model.layers:
            layer.trainable = trainable
        
        print('Resnet50 for Perceptual loss:')
        model.summary()
        return model

    def l2_normalize(self, x):
        """Apply L2 Normalization

        Args:
            x (tensor): output of convolution layer
        """        
        return K.l2_normalize(x, 0)

    def l2_normalize_output_shape(self, input_shape):
        return input_shape

    def global_average_pooling(self, x):
        """Apply global average pooling

        Args:
            x (tensor): output of convolution layer
        """
        x = GlobalAveragePooling2D()(x)
        return x

    def global_average_pooling_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def build_comp_attr_model(self, input_shape, trainable=False):
        _input = Input(shape=input_shape)
        resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=_input)
        activation_layers = []
        layers = resnet.layers
        for layer in layers:
            if 'activation' in layer.name:
                activation_layers.append(layer)

        activations = 0
        activations_gap_plus_lastactivation_gap_l2 = []
        # Create GAP layer for the activation layer at the end of each ResNet block, except for last one
        nlayers = len(activation_layers) - 1
        for i in range(1, nlayers):
            layer = activation_layers[i]
            # three activations per block, select only the last one
            if layer.output_shape[-1] > activation_layers[i - 1].output_shape[-1]:
                # print(layer.name, layer.input_shape, layer.output_shape[-1], activation_layers[i - 1].output_shape[-1])
                activations += layer.output_shape[-1]
                _out = Lambda(self.global_average_pooling,
                            output_shape=self.global_average_pooling_output_shape, name=layer.name + '_gap')(layer.output)
                activations_gap_plus_lastactivation_gap_l2.append(_out)

        print("sum of all activations should be 13056: {}".format(activations))

        last_layer_output = GlobalAveragePooling2D()(activation_layers[-1].output)

        last_layer_output = Lambda(self.l2_normalize, output_shape=self.l2_normalize_output_shape,
                                name=activation_layers[-1].name+'_gap')(last_layer_output)

        activations_gap_plus_lastactivation_gap_l2.append(last_layer_output)

        merged = Concatenate(axis=1)(activations_gap_plus_lastactivation_gap_l2)
        print("merged shape should be (?, 15104): ", merged.shape)
        merged = Lambda(self.l2_normalize, output_shape=self.l2_normalize_output_shape, name='merge')(merged)

        # create an output for each attribute
        outputs = []

        attrs = [k for k in self.loss_weights]
        for attr in attrs:
            outputs.append(Dense(1, kernel_initializer='glorot_uniform', activation='tanh', name=attr)(merged))

        non_negative_attrs = []
        for attr in non_negative_attrs:
            outputs.append(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid', name=attr)(merged))

        comp_attr_model = Model(inputs=_input, outputs=outputs)
        if self.comp_attrs_weights_path:
            print("comp_attrs_weights_path loaded")
            comp_attr_model.load_weights(self.comp_attrs_weights_path)
        else:
            print("comp_attrs_weights_path required for training on attributes")

        for layer in comp_attr_model.layers:
            comp_attr_model.trainable = trainable

        print('Resnet50 for Compositional Attributes loss:')
        comp_attr_model.summary()
        
        return comp_attr_model
              

    def mse_loss(self, y_true, y_pred):
        loss = K.mean(K.square(y_true - y_pred))
        return loss

    def build_model(self):
        # Calculate output shape of the Discriminator (PatchGAN)
        patch = int(self.img_size / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in first conv layer of generator and discriminator
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss weight, same as in orig paper
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss weight .5 lambda for monet and flower in orig paper
        self.lambda_feature = 1.0
        self.lambda_comp_attrs = 10

        #Optimizer
        optimizer = Adam(lr=self.base_lr, beta_1=self.beta_1)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Build the Perceptual Model
        if self.add_perceptual_loss:        
            self.model_perceptual = self.build_perceptual_model(input_shape=self.img_shape)

        # Build the Composition Attributes Model
        if self.target_attr_values:
            self.model_comp_attrs = self.build_comp_attr_model(input_shape=self.img_shape)

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Perceptual Feature Loss
        if self.add_perceptual_loss:
            percept_A = self.model_perceptual([img_A])
            percept_B = self.model_perceptual([img_B])
            percept_reconstr_A = self.model_perceptual([reconstr_A])
            percept_reconstr_B = self.model_perceptual([reconstr_B])

        # Compositional Attributes
        if self.target_attr_values:
            comp_attrs_A = self.model_comp_attrs(fake_A)
            comp_attrs_B = self.model_comp_attrs(fake_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Create Outputs Array
        outputs = [valid_A, valid_B, # d_A(g_BA(img_B), d_B(g_AB(img_A))
                    reconstr_A, reconstr_B,  # g_BA(g_AB(img_A)), g_AB(g_BA(img_B))
                    img_A_id, img_B_id] # g_BA(img_A), g_AB(img_B)
                    
        loss = ['mse', 'mse',
                'mae', 'mae',
                'mae', 'mae']

        loss_weights = [1, 1,
                        self.lambda_cycle, self.lambda_cycle,
                        self.lambda_id, self.lambda_id]

        
        if self.add_perceptual_loss:
            outputs.extend([percept_A, percept_B,
                           percept_reconstr_A, percept_reconstr_B,
                           percept_reconstr_A, percept_reconstr_B])
            loss.extend(['mse', 'mse',
                        'mse', 'mse',
                        'mse', 'mse'])
            loss_weights.extend([self.lambda_feature, self.lambda_feature,
                                self.lambda_feature, self.lambda_feature,
                                self.lambda_feature, self.lambda_feature])

        if self.target_attr_values:
            len_attrs = len(self.target_attr_values)

            attr_outputs = []
            for i in range(len_attrs):
                attr_outputs.extend([comp_attrs_A[i], comp_attrs_B[i]])
            outputs.extend(attr_outputs)
            
            loss.extend(['mse' for i in range(len_attrs * 2)])
            loss_weights.extend([self.lambda_comp_attrs for i in range(len_attrs * 2)])

        print('model outputs:', len(outputs))

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=outputs) 

        if self.weights_path:
            self.model.load_weights(self.weights_path)

        self.combined.compile(loss=loss,
                            loss_weights=loss_weights,
                            optimizer=optimizer)
