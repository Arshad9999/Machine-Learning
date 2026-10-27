import ast
import pathlib
import unittest

import numpy as np

from ForwardPropagation import forward_propagation


class ForwardPropagationTests(unittest.TestCase):
    def test_zero_weights_produce_half_probability(self):
        output = forward_propagation(np.array([[1, 2], [3, 4]]), np.zeros((2, 3)), np.zeros((3, 1)))
        np.testing.assert_allclose(output, [[0.5], [0.5]])

    def test_nonzero_hidden_layer_reaches_output(self):
        # sigmoid(0) = 0.5, then sigmoid(0.5 + 0.5) = sigmoid(1).
        output = forward_propagation(np.zeros((1, 2)), np.ones((2, 2)), np.ones((2, 1)))
        np.testing.assert_allclose(output, [[1 / (1 + np.exp(-1))]])

    def test_incompatible_dimensions_raise(self):
        with self.assertRaises(ValueError):
            forward_propagation(np.ones((1, 2)), np.ones((3, 2)), np.ones((2, 1)))

    def test_all_lab_files_parse_as_python(self):
        for path in pathlib.Path(__file__).resolve().parents[1].glob("*.py"):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(), filename=path.name)


if __name__ == "__main__":
    unittest.main()
