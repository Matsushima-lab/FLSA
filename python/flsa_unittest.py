import unittest
import sys
import numpy as np

import flsa_dp


class ReturnTest(unittest.TestCase):
    def test_easy_case(self):
        beta = flsa_dp.main([0, 1], 0.5)
        self.assertTrue(beta == [0.5, 0.5])

        args = sys.argv  # args: input, lam, output

        fi = open(args[0], "r")
        fo = open(args[2], "r")

        input_data = fi.read()
        output_data = fo.read()

        fi.close()
        fo.close()

        beta = flsa_dp.main(np.array(input_data), args[1], "squared")
        self.assertTrue(np.testing.assert_array_equal(
            beta, np.array(output_data)))

    def test_from_dp_cpp(self):
        beta = flsa_dp.main(self.input_data, self.lam)
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
