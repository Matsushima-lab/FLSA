import unittest
import scipy
import copy
import numpy as np
import os
import sys
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))
from clm_flsa_dp import DeltaCLM

EPS = 1e-5


class TestClmflsadp(unittest.TestCase):
    b_q_list = [-np.inf, -1, 0, 1, np.inf]


    def test_constructor(self):
        delta = DeltaCLM(3, self.b_q_list)
        assert(delta.coef_list == [[0, 0, 1, 1, 0, 1]])
        assert(delta.knots == [-np.inf, np.inf])
        assert(delta.basis_function_list[0](-np.inf) == 0)
        assert(delta.basis_function_list[4](np.inf) == -1)
        assert(delta.basis_function_list[1](0) == -1/(np.e + 1))
        assert(delta.basis_function_list[2](0) == -0.5)

    def test_first_calc_derivative_at(self):
        delta = DeltaCLM(2, self.b_q_list)
        initv = delta.calc_derivative_at(-np.inf, 0)
        assert(initv==-1)
        v=initv
        for b in range(-100,100,10):
            newv = delta.calc_derivative_at(b, 0)
            assert(delta.calc_derivative_at(b, 0) >= v)
            v = newv
        lastv = delta.calc_derivative_at(np.inf, 0)
        assert(lastv == 1)
        assert(lastv>=v)

    def test_find_tangency(self):
        delta = DeltaCLM(2, self.b_q_list)
        assert(delta.find_tangency(-1)==-np.inf)
        assert(delta.find_tangency(-2)==-np.inf)
        assert(abs(delta.find_tangency(0)+0.5) < EPS)
        assert(delta.find_tangency(1)==np.inf)
        assert(delta.find_tangency(2)==np.inf)
        assert(delta.find_tangency(-0.1) < delta.find_tangency(0.1))

    def test_forward(self):
        delta = DeltaCLM(2, self.b_q_list)
        delta = delta.forward(0.1, 4)
        assert(delta.find_tangency(0.1)==np.inf)
        delta = delta.forward(0.1, 1)
        print(vars(delta))



if __name__ == '__main__':
    unittest.main()
