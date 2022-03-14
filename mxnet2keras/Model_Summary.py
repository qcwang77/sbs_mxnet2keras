"""
Script that load and visualize the input Mxnet model.
Help users' to build corresponding model architecture in keras.
"""


import mxnet as mx


class ModelSummary:
    """
    The ModelSummary Class generate Mxnet model summary and visualizations.
    Help users' to build corresponding model architecture in keras.

    :param model_prefix: The prefix of the input mxnet model
    :param epoch: The number of epoch of the input mxnet model
    """

    def __init__(self, model_prefix, epoch):
        self.model_prefix = model_prefix
        self.epoch = epoch
        self.symbol = mx.model.load_checkpoint(model_prefix, epoch)[0]
        self.arg_params = mx.model.load_checkpoint(model_prefix, epoch)[1]
        self.aux_params = mx.model.load_checkpoint(model_prefix, epoch)[2]

    def plot(self, type='pdf', shape=None):
        """
        generate a graphical visualization of the mxnet model architecture.
        Saving the visualization in user-specified format in local directory.

        :param type: string that specify the plot output type
        :param shape: shape of model input data
        """
        graph = mx.viz.plot_network(self.symbol, shape=shape)
        graph.format = type
        graph.render('graph')
        print('graph.{} can be accsssed within current directory'.format(type))

    def summary(self, shape=None):
        """
        generate a printed summary of the mxnet input model architecture.

        :param shape: shape of model input data
        """
        summary_output = mx.viz.print_summary(self.symbol, shape=shape)
        return summary_output

    def get_layer_names(self):
        """
        generate a list of layer names of the input model.

        :return: target_call: list of model name str
        """
        weight_name = [key for key, _ in self.arg_params.items()]
        # extract layer name from weight name
        layer_name = list(map(lambda key: key.rsplit("_", 1)[0], weight_name))
        layer_name_cleaned = list(set(layer_name))
        return layer_name_cleaned

    def filtered_layer_names(self, layer_type='all'):
        """
        generate a list of a type of layer names that user specified.

        :param layer_type: user specified typeo of layer str
        :return: target_call: list of model name str
        """
        # currently only support four common types of layer: conv, bn, reLU, fc
        batchnorm_layer_name = list(
            filter(
                lambda key: 'batchnorm' in key,
                self.get_layer_names()))
        convolution_layer_name = list(
            filter(
                lambda key: 'conv' in key,
                self.get_layer_names()))
        reLu_layer_name = list(
            filter(
                lambda key: 'relu' in key,
                self.get_layer_names()))
        fc_layer_name = list(
            filter(
                lambda key: 'fc' in key,
                self.get_layer_names()))

        if layer_type == 'all':
            result = []
            result.append(batchnorm_layer_name)
            result.append(convolution_layer_name)
            result.append(reLu_layer_name)
            result.append(fc_layer_name)
            return result
        elif layer_type in ('batchnorm', 'bn', 'Batchnorm'):
            return batchnorm_layer_name
        elif layer_type in ('convolution', 'conv', 'Convolution'):
            return convolution_layer_name
        elif layer_type in ('relu', 'ReLu', 'reLu'):
            return reLu_layer_name
        elif layer_type in ('fc', 'fully connected', 'Fully Connected'):
            return fc_layer_name
        else:
            print('No such layers found in the network, please check again!')
