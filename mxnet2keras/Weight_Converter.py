"""
Script that convert layer
"""

import mxnet as mx
from mxnet2keras.Model_Summary import ModelSummary
from mxnet2keras.Weight_Transfer_Functions import transfer_batchnorm_weight
from mxnet2keras.Weight_Transfer_Functions import transfer_conv_weight
from mxnet2keras.Weight_Transfer_Functions import transfer_relu_weight
from mxnet2keras.Weight_Transfer_Functions import transfer_fc_weight


class WeightConvert:

    def __init__(self, model_prefix, epoch, keras_net):
        self.keras_net = keras_net
        self.model_prefix = model_prefix
        self.epoch = epoch
        self.symbol = mx.model.load_checkpoint(model_prefix, epoch)[0]
        self.arg_params = mx.model.load_checkpoint(model_prefix, epoch)[1]
        self.aux_params = mx.model.load_checkpoint(model_prefix, epoch)[2]
        self.layer_name = ModelSummary(model_prefix, epoch).get_layer_names()
        self.bn = ModelSummary(model_prefix, epoch).filtered_layer_names()[0]
        self.conv = ModelSummary(model_prefix, epoch).filtered_layer_names()[1]
        self.reLu = ModelSummary(model_prefix, epoch).filtered_layer_names()[2]
        self.fc = ModelSummary(model_prefix, epoch).filtered_layer_names()[3]

    def load_single_weight(self, layer_name):
        if layer_name in self.bn:
            transfer_batchnorm_weight(
                self.keras_net,
                layer_name,
                self.arg_params,
                self.aux_params)
        elif layer_name in self.conv:
            transfer_conv_weight(self.keras_net, layer_name, self.arg_params)
        elif layer_name in self.reLu:
            transfer_relu_weight(self.keras_net, layer_name, self.arg_params)
        elif layer_name in self.fc:
            transfer_fc_weight(self.keras_net, layer_name, self.arg_params)
        else:
            print('Please Check the layer name!')

    def load_type_weight(self, layer_type):
        if layer_type in ('batchnorm', 'bn', 'Batchnorm'):
            for layer_name in self.bn:
                transfer_batchnorm_weight(
                    self.keras_net,
                    layer_name,
                    self.arg_params,
                    self.aux_params)
        elif layer_type in ('convolution', 'conv', 'Convolution'):
            for layer_name in self.conv:
                transfer_conv_weight(
                    self.keras_net, layer_name, self.arg_params)
        elif layer_type in ('relu', 'ReLu', 'reLu'):
            for layer_name in self.reLu:
                transfer_relu_weight(
                    self.keras_net, layer_name, self.arg_params)
        elif layer_type in ('fc', 'fully connected', 'Fully Connected'):
            for layer_name in self.fc:
                transfer_fc_weight(self.keras_net, layer_name, self.arg_params)
        else:
            print('No such layers found in the network, please check again!')

    def load_all_weight(self):
        self.load_type_weight('batchnorm')
        self.load_type_weight('convolution')
        self.load_type_weight('relu')
        self.load_type_weight('fc')
