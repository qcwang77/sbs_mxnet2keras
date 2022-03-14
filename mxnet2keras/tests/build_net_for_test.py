"""
build a corresponding keras model architecture.
Helper function for test_Weight_Converter.py
"""
import keras
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Lambda
from keras.layers import Dense, LeakyReLU, Dropout, Activation, Add
from keras.activations import relu, softmax


def build_net():
    """
    build a corresponding keras model architecture for testing purpose
    :return model: keras model that follows the mxnet model architecture
    """

    input_data = keras.layers.Input(shape=(32, 1000, 1))
    x1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                name='conv-0', use_bias=True)(input_data)
    x1 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-0')(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                name='conv-1', use_bias=True)(x1)
    x1 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-1')(x1)
    x1 = LeakyReLU()(x1)
    x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x1 = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                name='conv-2', use_bias=True)(x1)
    x1 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-2')(x1)
    x1 = LeakyReLU()(x1)

    x1 = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                name='conv-3', use_bias=True)(x1)
    x1 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-3')(x1)
    x1 = LeakyReLU()(x1)
    x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)

    x2 = Conv2D(256, (1, 1), strides=(1, 1), padding="same",
                name='conv-4-1-1x1', use_bias=True)(x1)
    x2 = LeakyReLU()(x2)

    x2 = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                name='conv-4', use_bias=True)(x2)
    x2 = LeakyReLU()(x2)

    x2 = Conv2D(512, (1, 1), strides=(1, 1), padding="same",
                name='conv-4-2-1x1', use_bias=True)(x2)
    x2 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-4')(x2)
    x2 = LeakyReLU()(x2)

    x2 = Conv2D(256, (1, 1), strides=(1, 1), padding="same",
                name='conv-5-1-1x1', use_bias=True)(x2)
    x2 = LeakyReLU()(x2)

    x2 = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                name='conv-5', use_bias=True)(x2)
    x2 = LeakyReLU()(x2)

    x2 = Conv2D(512, (1, 1), strides=(1, 1), padding="same",
                name='conv-5-2-1x1', use_bias=True)(x2)
    x2 = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-5')(x2)
    x2 = LeakyReLU()(x2)

    x = Add()([x1, x2])

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(256, (1, 1), strides=(1, 1), padding="same",
               name='conv-6-1-1x1', use_bias=True)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, (4, 1), strides=(1, 1), padding="valid",
               name='conv-6', use_bias=True)(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, (1, 1), strides=(1, 1), padding="same",
               name='conv-6-2-1x1', use_bias=True)(x)
    x = BatchNormalization(
        axis=-1,
        epsilon=0.0010000000474974513,
        momentum=0.899999976,
        name='batchnorm-6')(x)
    x = LeakyReLU()(x)

    x = Dropout(0.3)(x)
    x = Lambda((lambda x: x[:, 0, :, :]))(x)

    x = Dense(128, use_bias=True, name='seq-fc')(x)
    x = Activation(relu)(x)

    x = Dense(6426, use_bias=True, name='pred_fc')(x)
    x = Activation(softmax)(x)

    model = keras.models.Model(inputs=input_data, outputs=x)
    model.compile(optimizer='Adam', loss='mse')

    return model
