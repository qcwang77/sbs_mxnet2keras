"""
Unittests for Model_Summary.py
"""

import unittest
import os.path
import mxnet2keras
from mxnet2keras.Model_Summary import ModelSummary


data_path = os.path.join(mxnet2keras.__path__[0], 'data')


class TestModelSummary(unittest.TestCase):
    """
    Test cases for the class ModelSummary
    """

    def test_plot_sum(self):
        """
        Test case for the function plot() and sum().
        Test if the graphical visualization is generated.
        """

        ocr_model = ModelSummary(os.path.join(data_path, 'cnocr-v1.2.0-conv-lite-fc'), 25)
        ocr_model.summary()
        ocr_model.plot('pdf')
        self.assertTrue(os.path.isfile('./graph.pdf'))

    def test_get_layer_names(self):
        """
        Test case for the function get_layer_names().
        Test if the output has all the layer names listed.
        """

        ocr_model = ModelSummary('cnocr-v1.2.0-conv-lite-fc', 25)
        self.assertEqual(len(ocr_model.get_layer_names()), 22)

    def test_filtered_layer_names(self):
        """
        Test case for the function filtered_layer_names().
        Test if the output correctly filtered all layer names.
        """

        ocr_model = ModelSummary('cnocr-v1.2.0-conv-lite-fc', 25)
        self.assertEqual(len(ocr_model.filtered_layer_names()[0]), 7)
        self.assertEqual(len(ocr_model.filtered_layer_names()[1]), 13)
        self.assertEqual(len(ocr_model.filtered_layer_names()[2]), 0)
        self.assertEqual(len(ocr_model.filtered_layer_names()[3]), 2)

if __name__ == '__main__':
    unittest.main()