from __future__ import annotations
from abc import abstractmethod
from colorsys import yiq_to_rgb
from typing import overload
import numpy as np
from abc import ABC, abstractmethod

from turtle import forward, right


class DeltaFunc(ABC):

    basis_function_list = [] # bases for delta'

    def __init__(self, coef_list) -> None:
        """init function for
            delta'(b)のi番目のknot区間について
            delta'(b) = a_b_list[i] * b + a_c_list[i]

        Args:
            bm (float): b^-. This variable will be used in backward process.
            bp (float): b^+. This variable will be used in backward process.
            knots (List(float)): The interval of delta function between two adjacent knots can be expressed by delta(b) = sum(coef * basis_function(b)).
                coef and basis function is an element of coef_list and basis_function_list.
            coef_list (List(float)): The length of coef_list is len(knots) + 1
        """
        self.coef_list = coef_list

    def backward(self, next_beta):
        """
        Args:
        Return:beta
        """
        return max(min(next_beta, self.bp), self.bm)
        """
        self.b is a float or a set of float that satisfies delta' = +-lambda
        Calculated in the process of "forward" method
        """

    def forward(self, lamb: float, next_yi: float) -> DeltaFunc:
        """
        Compute next delta(b) as min_b' delta(b') + loss(b,yi) +  lambda |b'-b|

        Args:
        Return; DeltaFunc
        """
        self.bm = self.find_tangency(-lamb)
        self.bp = self.find_tangency(lamb)
        next_f = self.overwrite(self.bm, self.bp, lamb)
        next_delta = self.add_loss(next_f, next_yi)
        return self.return_instance(next_delta)

    @abstractmethod
    def find_tangency(self, g):
        '''
        find a knot "t" s.t. t = argmin(delta(t) - g*t)
        '''
        pass

    @abstractmethod
    def overwrite(self, left_new_knot, right_new_knot, const):
        '''
        derive f' from previous delta'
        '''
        pass

    @abstractmethod
    def add_loss(self, next_f, next_yi):
        '''
        calculate e' + f'
        '''
        pass

    @abstractmethod
    def get_constant_f(self, x):
        '''
        return the expression for the leftmost or rightmost interval of f
        '''
        pass

    @abstractmethod
    def calc_inverse_spline(self, t, d):
        '''
        inverse function for delta'
        return "b" such that delta'(b) = d
        "t" indicates the interval in which such "b" exists.
        '''
        pass

    @abstractmethod
    def calc_derivative_at(self, b, t):
        '''
        return the value of delta' for a given "t"
        '''
        pass

    @abstractmethod
    def return_instance(self, next_delta):
        pass


class DeltaLogistic(DeltaFunc):
    basis_function_list = [lambda x: -1/(np.exp(x) + 1), lambda x: 1/(np.exp(-x) + 1), lambda x: 1]
    tangency_intervals = []

    def __init__(self, knots, coef_list):
        super().__init__(coef_list)
        self.knots = knots or [-np.inf, np.inf]

    def find_tangency(self, g):
        for t in range(len(self.knots)-1):
            if self.calc_derivative_at(self.knots[t+1], t) - g > 0:
                self.tangency_intervals.append(t+1)
                return self.calc_inverse_spline(t,g)
        self.tangency_intervals.append(len(self.knots))
        return np.inf

    def overwrite(self, left_new_knot, right_new_knot, const):
        tmp_knots = self.knots[self.tangency_intervals[0]:self.tangency_intervals[1]]
        tmp_coef_list = self.coef_list[self.tangency_intervals[0]-1:self.tangency_intervals[1]]

        if left_new_knot != -np.inf:
            tmp_knots = [-np.inf, left_new_knot] + tmp_knots
            tmp_coef_list = [self.get_constant_f(-const)] + tmp_coef_list
        else:
            tmp_knots = [-np.inf] + tmp_knots

        if right_new_knot != np.inf:
            tmp_knots = tmp_knots + [right_new_knot, np.inf]
            tmp_coef_list = tmp_coef_list + [self.get_constant_f(const)]
        else:
            tmp_knots = tmp_knots + [np.inf]

        return tmp_knots, tmp_coef_list

    def add_loss(self, next_f, next_yi):
        for coefs in next_f[1]:
            if(next_yi == 1):
                coefs[0] += 1
            else:
                coefs[1] += 1
        return next_f

    def get_constant_f(self, x):
        return [0, 0, x]

    def calc_inverse_spline(self, t, d):
        if (0 < self.coef_list[t][2] - d < self.coef_list[t][0]) or (-self.coef_list[t][1] < self.coef_list[t][2] - d < 0):
            return np.log((self.coef_list[t][0] - self.coef_list[t][2] + d) / (self.coef_list[t][1] + self.coef_list[t][2] - d))
        elif (self.coef_list[t][2] - d > self.coef_list[t][0]):
            return -np.inf
        else: #not necessary?
            return np.inf

    def calc_derivative_at(self, b, t):
        return sum(
            [
                c * bf(b)
                for bf, c in zip(self.basis_function_list, self.coef_list[t])
            ]
        )

    def return_instance(self, next_delta):
        return DeltaLogistic(knots=next_delta[0], coef_list=next_delta[1])


class DeltaSquared(DeltaFunc):
    basis_function_list = [lambda x: x, lambda x: 1]
    tangency_intervals = []

    def __init__(
        self,
        knots,
        coef_list,
    ):
        """init function for
            delta'(b)のi番目のknot区間について
            delta'(b) = a_b_list[i] * b + a_c_list[i]

        Args:
            bm (float): b^-. This variable will be used in backward process.
            bp (float): b^+. This variable will be used in backward process.
            knots (List(float)): The interval of delta function between two adjacent knots can be expressed by delta(b) = sum(coef * basis_function(b)).
                coef and basis function is an element of coef_list and basis_function_list.
            coef_list (List(float)): The length of coef_list is len(knots) + 1
        """
        super().__init__(coef_list=coef_list)
        self.knots = knots

    def find_tangency(self, g):
        for t in range(len(self.knots)):
            if self.calc_derivative_at(self.knots[t], t) - g > 0:
                self.tangency_intervals.append(t)
                return self.calc_inverse_spline(t,g)
        self.tangency_intervals.append(len(self.knots))
        return self.calc_inverse_spline(len(self.knots), g)

    def overwrite(self, left_new_knot, right_new_knot, lamb):
        tmp_knots = [left_new_knot] + self.knots[self.tangency_intervals[0]:self.tangency_intervals[1]] + [right_new_knot]
        tmp_coef_list = [self.get_constant_f(-lamb)] + self.coef_list[self.tangency_intervals[0]:self.tangency_intervals[1]+1] + [self.get_constant_f(lamb)]

        return tmp_knots, tmp_coef_list

    def add_loss(self, next_f, next_yi):
        for coefs in next_f[1]:
            coefs[0] += 1
            coefs[1] -= next_yi

        return next_f

    def get_constant_f(self, x):
        return [0, x]

    def calc_inverse_spline(self, t, d):
        assert self.coef_list[t][0] != 0
        return (d - self.coef_list[t][1]) / self.coef_list[t][0]

    def calc_derivative_at(self, b, t):
        return sum(
            [
                c * bf(b)
                for bf, c in zip(self.basis_function_list, self.coef_list[t])
            ]
        )

    def return_instance(self, next_delta):
        return DeltaSquared(knots = next_delta[0], coef_list = next_delta[1])


def solver(y: np.array, lamb: float, loss: str = "squared") -> np.array:
    n = y.size
    delta = [None] * n
    beta = np.zeros(n)

    if loss == "squared":
        delta[0] = DeltaSquared(knots=[], coef_list=[[1, -y[0]]])
    elif loss == "logistic":
        if y[0] == 1:
            delta[0] = DeltaLogistic(knots=[-np.inf, np.inf], coef_list=[[1, 0, 0]])
        else:
            delta[0] = DeltaLogistic(knots=[-np.inf, np.inf], coef_list=[[0, 1, 0]])
    
    for i in range(n - 1):
        delta[i + 1] = delta[i].forward(
            lamb, y[i + 1])
        print(f"delta_squared[{i + 1}]:", vars(delta[i + 1]))
    beta[n - 1] = delta[n - 1].find_tangency(0)
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta[i-1].backward(next_beta=beta[i])

    return beta


if __name__ == "__main__":
    beta1 = solver(np.array([0,1]), 0.5)
    beta2 = solver(np.array([-1, -1, 1, -1, 1, 1]), 0.5, "logistic")
    print(beta1)
    print(beta2)