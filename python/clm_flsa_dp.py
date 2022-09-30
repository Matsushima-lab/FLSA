from flsa_dp import DeltaFunc
import numpy as np
import copy


class DeltaCLM(DeltaFunc):

    def __init__(self, y=None, b_q_list=[]):
        self.coef_list = [0] * (len(b_q_list)) + [1]
        self.coef_list[y] = 1
        self.coef_list[y - 1] = 1
        self.knots = [-np.inf, np.inf]
        self.tangency_intervals = []
        self.b_q_list = b_q_list
        self.basis_function_list = [
            lambda x: -1/(np.exp(bq - x) + 1) for bq in b_q_list] + [lambda x: 1]

    def find_tangency(self, g):
        for t in range(len(self.knots)-1):
            if self.calc_derivative_at(self.knots[t+1], t) - g > 0:
                self.tangency_intervals.append(t+1)
                return self.calc_inverse_spline(t, g)
        self.tangency_intervals.append(len(self.knots))
        return np.inf

    def copy(self):
        new = DeltaCLM()
        new.knots = copy.copy(self.knots)
        new.coef_list = copy.deepcopy(self.coef_list)
        new.tangency_intervals = copy.copy(self.tangency_intervals)
        return new

    def overwrite(self, left_new_knot, right_new_knot, lamb):
        tmp_knots = self.knots[self.tangency_intervals[0]:self.tangency_intervals[1]]
        tmp_coef_list = self.coef_list[self.tangency_intervals[0] -
                                       1:self.tangency_intervals[1]]

        if left_new_knot != -np.inf:
            tmp_knots = [-np.inf, left_new_knot] + tmp_knots
            tmp_coef_list = [self.get_constant_f(-lamb)] + tmp_coef_list
        else:
            tmp_knots = [-np.inf] + tmp_knots

        if right_new_knot != np.inf:
            tmp_knots = tmp_knots + [right_new_knot, np.inf]
            tmp_coef_list = tmp_coef_list + [self.get_constant_f(lamb)]
        else:
            if tmp_knots[-1] != np.inf:
                tmp_knots = tmp_knots + [np.inf]

        self.knots = tmp_knots
        self.coef_list = tmp_coef_list

        self.tangency_intervals = []

        return self

    def add_loss(self, next_yi):
        for coef in self.coef_list:
            coef[next_yi - 1] += 1
            coef[next_yi] += 1
            coef[-1] += 1
        return self

    def get_constant_f(self, x):
        return [0] * len(self.b_q_list) + [x]

    def calc_inverse_spline(self, t, d):
        self.

    def calc_derivative_at(self, b, t):
        pass

    def return_instance(self, next_delta):
        return DeltaCLM(knots=next_delta[0], coef_list=next_delta[1])
