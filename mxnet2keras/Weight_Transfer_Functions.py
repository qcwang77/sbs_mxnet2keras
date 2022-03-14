"""
Script that defines weight transfer functions.
This script is used as a component of Weight Converter
"""

import numpy as np


def transfer_fc_weight(keras_net, layer_name, arg_params):
    """
    transfer weight for a specific fully connected layer.

    :param keras_net: user defined keras model architecture
    :param layer_name: layer name str
    :param arg_params: mxnet model weight file arg_params
    """

    mxnet_weight = arg_params[layer_name + "_weight"].asnumpy()
    mxnet_bias = arg_params[layer_name + "_bias"].asnumpy()

    keras_layer = keras_net.get_layer(layer_name)
    keras_weight, keras_bias = keras_layer.get_weights()

    keras_bias = mxnet_bias

    for i in range(mxnet_weight.shape[0]):
        keras_weight[:, i] = mxnet_weight[i, :]

    keras_layer.set_weights([keras_weight, keras_bias])


def transfer_batchnorm_weight(keras_net, layer_name, arg_params, aux_params):
    """
    transfer weight for a specific batchnorm layer.

    :param keras_net: user defined keras model architecture
    :param layer_name: layer name str
    :param arg_params: mxnet model saved weight file arg_params
    :param aux_params: mxnet model saved weight file aux_params
    """

    gamma = arg_params[layer_name + "_gamma"].asnumpy()
    beta = arg_params[layer_name + "_beta"].asnumpy()
    moving_var = aux_params[layer_name + "_moving_var"].asnumpy()
    moving_mean = aux_params[layer_name + "_moving_mean"].asnumpy()

    keras_layer = keras_net.get_layer(layer_name)
    keras_layer.set_weights([gamma, beta, moving_mean, moving_var])


def transfer_relu_weight(keras_net, layer_name, arg_params):
    """
    transfer weight for a specific reLu layer.

    :param keras_net: user defined keras model architecture
    :param layer_name: layer name str
    :param arg_params: mxnet model saved weight file arg_params
    """

    gamma = arg_params[layer_name + "_gamma"].asnumpy()

    keras_layer = keras_net.get_layer(layer_name)
    keras_weight = np.reshape(gamma, (1, 1, 1, -1))
    keras_layer.set_weights(keras_weight)


def transfer_conv_weight(keras_net, layer_name, arg_params):
    """
    transfer weight for a specific convolution layer.

    :param keras_net: user defined keras model architecture
    :param layer_name: layer name str
    :param arg_params: mxnet model saved weight file arg_params
    """

    mxnet_weight = arg_params[layer_name + "_weight"].asnumpy()
    mxnet_bias = arg_params[layer_name + "_bias"].asnumpy()

    keras_layer = keras_net.get_layer(layer_name)
    keras_weight, keras_bias = keras_layer.get_weights()

    shape1, shape2, shape3, shape4 = mxnet_weight.shape

    keras_bias = mxnet_bias

    for i in range(shape1):
        for j in range(shape2):
            for l in range(shape3):
                for k in range(shape4):
                    keras_weight[l, k, j, i] = mxnet_weight[i, j, l, k]

    keras_layer.set_weights([keras_weight, keras_bias])
