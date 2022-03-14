"""
Unittests for Weight_Converter.py
"""
import unittest
import os.path
import mxnet2keras
from mxnet2keras.Weight_Converter import WeightConvert
from mxnet2keras.tests.build_net_for_test import build_net

data_path = os.path.join(mxnet2keras.__path__[0], 'data')


class TestModelWeightConverter(unittest.TestCase):
    """
    Test cases for the class WeightConverter
    """
    def setUp(self):
        test_model = build_net()
        self.loadweight = WeightConvert(os.path.join(
            data_path, 'cnocr-v1.2.0-conv-lite-fc'), 25, test_model)

    def test_load_single_weight(self):
        """
        Test case for the function load_single_weight().
        Test if the bias loaded correctly for single conv layer.
        """
        self.loadweight.load_single_weight('conv-1')
        bias_mx = self.loadweight.arg_params['conv-1' + "_bias"].asnumpy()
        layer = self.loadweight.keras_net.get_layer('conv-1')
        bias_keras = layer.get_weights()[1]
        self.assertEqual(max(bias_mx - bias_keras), 0)

    def test_load_type_weight(self):
        """
        Test case for the function load_type_weight().
        Test if the bias loaded correctly for all conv layers.
        """
        self.loadweight.load_type_weight('conv')
        bias_mx = self.loadweight.arg_params['conv-2' + "_bias"].asnumpy()
        layer = self.loadweight.keras_net.get_layer('conv-2')
        bias_keras = layer.get_weights()[1]
        self.assertEqual(max(bias_mx - bias_keras), 0)

    def test_load_all_weight(self):
        """
        Test case for the function load_type_weight().
        Test if the gamma, beta loaded correctly for all bn layers.
        """
        self.loadweight.load_all_weight()
        gamma_mx = self.loadweight.arg_params['batchnorm-0' +
                                              "_gamma"].asnumpy()
        beta_mx = self.loadweight.arg_params['batchnorm-0' + "_beta"].asnumpy()
        layer = self.loadweight.keras_net.get_layer('batchnorm-0')
        gamma_keras, beta_keras = layer.get_weights()[:2]
        self.assertEqual(max(gamma_mx - gamma_keras), 0)
        self.assertEqual(max(beta_mx - beta_keras), 0)


if __name__ == '__main__':
    unittest.main()
