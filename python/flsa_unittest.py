import unittest, sys
import numpy as np

import flsa_dp

class ReturnTest(unittest.TestCase):
    def test_return(self):
        beta = flsa_dp.main(np.array([0, 1]), 0.5, "str")
        self.assertTrue(
            np.testing.assert_array_equal(beta, np.array([0.5, 0.5]))
        )

        args = sys.argv #args: input, lam, output
        beta = flsa_dp.main(np.array(args[0]), args[1], "str")
        self.assertTrue(
            np.testing.assert_array_equal(beta, np.array(args[2]))
        )


if __name__ == "__main__":
    unittest.main()
