"""
Script that convert layer
"""

import mxnet as mx
from mxnet2keras.Model_Summary import ModelSummary
from mxnet2keras.Weight_Transfer_Functions import transfer_batchnorm_weight, transfer_conv_weight, transfer_relu_weight, transfer_fc_weight


class WeightConvert:

    def __init__(self, model_prefix, epoch, keras_net):
        self.keras_net = keras_net
        self.model_prefix = model_prefix
        self.epoch = epoch
        self.symbol, self.arg_params, self.aux_params = mx.model.load_checkpoint(
            model_prefix, epoch)
        self.layer_name = ModelSummary(model_prefix, epoch).get_layer_names()
        self.batchnorm_layer_name, self.convolution_layer_name, self.reLu_layer_name, self.fc_layer_name = ModelSummary(
            model_prefix, epoch).filtered_layer_names()

    def load_single_weight(self, layer_name):
        if layer_name in self.batchnorm_layer_name:
            transfer_batchnorm_weight(
                self.keras_net,
                layer_name,
                self.arg_params,
                self.aux_params)
        elif layer_name in self.convolution_layer_name:
            transfer_conv_weight(self.keras_net, layer_name, self.arg_params)
        elif layer_name in self.reLu_layer_name:
            transfer_relu_weight(self.keras_net, layer_name, self.arg_params)
        elif layer_name in self.fc_layer_name:
            transfer_fc_weight(self.keras_net, layer_name, self.arg_params)
        else:
            print('Please Check the layer name!')

    def load_type_weight(self, layer_type):
        if layer_type in ('batchnorm', 'bn', 'Batchnorm'):
            for layer_name in self.batchnorm_layer_name:
                transfer_batchnorm_weight(
                    self.keras_net,
                    layer_name,
                    self.arg_params,
                    self.aux_params)
        elif layer_type in ('convolution', 'conv', 'Convolution'):
            for layer_name in self.convolution_layer_name:
                transfer_conv_weight(
                    self.keras_net, layer_name, self.arg_params)
        elif layer_type in ('relu', 'ReLu', 'reLu'):
            for layer_name in self.reLu_layer_name:
                transfer_relu_weight(
                    self.keras_net, layer_name, self.arg_params)
        elif layer_type in ('fc', 'fully connected', 'Fully Connected'):
            for layer_name in self.fc_layer_name:
                transfer_fc_weight(self.keras_net, layer_name, self.arg_params)
        else:
            print('No such layers found in the network, please check again!')

    def load_all_weight(self):
        self.load_type_weight('batchnorm')
        self.load_type_weight('convolution')
        self.load_type_weight('relu')
        self.load_type_weight('fc')
