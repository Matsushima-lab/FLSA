import unittest, sys
import numpy as np

import flsa_dp


class ReturnTest(unittest.TestCase):
    def test_return(self):
        beta = flsa_dp.main(np.array([0, 1]), 0.5, "squared")
        self.assertTrue(np.testing.assert_array_equal(beta, np.array([0.5, 0.5])))

        args = sys.argv  # args: input, lam, output

        fi = open(args[0], "r")
        fo = open(args[2], "r")

        input_data = fi.read()
        output_data = fo.read()

        fi.close()
        fo.close()

        beta = flsa_dp.main(np.array(input_data), args[1], "squared")
        self.assertTrue(np.testing.assert_array_equal(beta, np.array(output_data)))


if __name__ == "__main__":
    unittest.main()
