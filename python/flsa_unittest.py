import unittest
import sys
import numpy as np

import flsa_dp


class ReturnTest(unittest.TestCase):
    def test_easy_case(self):
        beta = flsa_dp.solver(np.array([0, 1]), 0.5)
        np.testing.assert_array_almost_equal(np.array(beta), np.array([0.5, 0.5]))

    def test_from_dp_cpp(self):
        beta = flsa_dp.solver(np.array(self.input_data), self.lam)
        np.testing.assert_array_almost_equal(
            np.array(beta), np.array(self.output_data), decimal=5
        )


if __name__ == "__main__":
    # arg: input, lam, output
    assert len(sys.argv) > 1
    args = sys.argv

    ReturnTest.input_data = []
    ReturnTest.output_data = []
    ReturnTest.lam = float(args[2])

    with open(args[1], "r") as fi:
        for input_str in fi.readlines():
            input_float = float(input_str)
            ReturnTest.input_data.append(input_float)
    with open(args[3], "r") as fo:
        for output_str in fo.readlines():
            output_float = float(output_str)
            ReturnTest.output_data.append(output_float)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)
