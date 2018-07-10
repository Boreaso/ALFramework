import keras.backend as K
import keras.layers as layers
from keras import Input, Model
from keras.layers import Activation, AveragePooling2D, \
    BatchNormalization, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    MaxPooling2D
from keras.optimizers import Adam

from models_keras.base_model import BaseModel
from models_keras.keras_metrics import f1score, precision, recall


class ResNetModel(BaseModel):
    """Build a vgg-16 like model with keras."""

    def _build_model(self):
        def identity_block(input_tensor, kernel_size, filters, stage, block):
            """The identity block is the block that has no conv layer at shortcut.

            # Arguments
                input_tensor: input tensor
                kernel_size: default 3, the kernel size of middle conv layer at main path
                filters: list of integers, the filters of 3 conv layer at main path
                stage: integer, current stage label, used for generating layer names
                block: 'a','b'..., current block label, used for generating layer names

            # Returns
                Output tensor for the block.
            """
            filters1, filters2, filters3 = filters
            if K.image_data_format() == 'channels_last':
                _bn_axis = 3
            else:
                _bn_axis = 1
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = Conv2D(filters2, kernel_size,
                       padding='same', name=conv_name_base + '2b')(x)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2c')(x)

            x = layers.add([x, input_tensor])
            x = Activation('relu')(x)
            return x

        def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
            """A block that has a conv layer at shortcut.

            # Arguments
                input_tensor: input tensor
                kernel_size: default 3, the kernel size of middle conv layer at main path
                filters: list of integers, the filters of 3 conv layer at main path
                stage: integer, current stage label, used for generating layer names
                block: 'a','b'..., current block label, used for generating layer names
                strides: Strides for the first conv layer in the block.

            # Returns
                Output tensor for the block.

            Note that from stage 3,
            the first conv layer at main path is with strides=(2, 2)
            And the shortcut should have strides=(2, 2) as well
            """
            filters1, filters2, filters3 = filters
            if K.image_data_format() == 'channels_last':
                _bn_axis = 3
            else:
                _bn_axis = 1
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = Conv2D(filters1, (1, 1), strides=strides,
                       name=conv_name_base + '2a')(input_tensor)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = Conv2D(filters2, kernel_size, padding='same',
                       name=conv_name_base + '2b')(x)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
            x = BatchNormalization(axis=_bn_axis, name=bn_name_base + '2c')(x)

            shortcut = Conv2D(filters3, (1, 1), strides=strides,
                              name=conv_name_base + '1')(input_tensor)
            shortcut = BatchNormalization(axis=_bn_axis, name=bn_name_base + '1')(shortcut)

            x = layers.add([x, shortcut])
            x = Activation('relu')(x)
            return x

        bn_axis = -1  # BN for channels.
        if self._feature_type == 'raw':
            # Input
            inputs = Input(shape=[96, 64, 1], name='input')

            # Stage 1
            x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

            # Stage 2
            x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
            x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

            # Stage 3
            x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
            x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

            # Stage 4
            x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

            # Stage 5
            x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
            x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

            x = AveragePooling2D((2, 2), name='avg_pool')(x)  # Changed from 7*7 to 2*2

            x = Flatten()(x)
            x = Dense(1, activation='sigmoid', name='fc')(x)

            # Create model.
            model = Model(inputs, x, name='resnet50')
        elif self._feature_type == 'embedding':
            inputs = Input(shape=self._embedding_size)
            x = GlobalMaxPooling2D()(inputs)
            x = Flatten()(x)
            x = Dense(1, activation='sigmoid', name='fc')(x)
            model = Model(inputs, x, name='resnet50')
        else:
            raise ValueError("Unknown feature type: %s." % self._feature_type)

        optimizer = Adam(lr=self._learning_rate)
        model.compile(
            optimizer=optimizer, loss='binary_crossentropy',
            metrics=['accuracy', precision, recall, f1score])

        # Init special field.
        self._embedding_output_layer_name = 'fc'
        self._classifier_input_layer_name = 'fc'
        self._classifier_output_layer_name = 'fc'
        self._gradients_layer_name = 'fc'

        return model
