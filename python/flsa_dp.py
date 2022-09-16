from __future__ import annotations
from abc import abstractmethod
from colorsys import yiq_to_rgb
from typing import overload
import numpy as np
from abc import ABC, abstractmethod

from turtle import forward


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
        find a knot "t" s.t. delta(t) - g*t = 0
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
    def calc_derivative_at(self, t):
        '''
        return the value of delta' for a given "t"
        '''
        pass

    @abstractmethod
    def return_instance(self, next_delta):
        pass


class DeltaLogistic(DeltaFunc):
    def __init__(self, knots, coef_list):
        super().__init__(coef_list)
        self.knots = knots or [-np.inf, np.inf]

    def find_tangency(self, g):
        pass

    def overwrite(self, left_new_knot, right_new_knot, const):
        return super().overwrite(left_new_knot, right_new_knot, const)

    def add_loss(self, next_f, next_yi):
        return super().add_loss(next_f, next_yi)

    def get_constant_f(self, x):
        return super().get_constant_f(x)

    def calc_inverse_spline(self, t, d):
        return super().calc_inverse_spline(t, d)

    def calc_derivative_at(self, t):
        return super().calc_derivative_at(t)

    def return_instance(self, next_delta):
        return DeltaLogistic(next_delta)


class DeltaSquared(DeltaFunc):
    basis_function_list = [lambda x: x, lambda x: 1]

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
                return self.calc_inverse_spline(t,g)

        return self.calc_inverse_spline(len(self.knots), g)

    def overwrite(self, left_new_knot, right_new_knot, lamb):
        left_most = None
        right_most = None
        for t in range(len(self.knots)):
            if abs(self.calc_derivative_at(left_new_knot, t) + lamb) < 1e-7:
                left_most = t
        if left_most == None: 
            left_most = len(self.knots)
            right_most = len(self.knots)
        else:
            for t in range(left_most, len(self.knots)):
                if abs(self.calc_derivative_at(right_new_knot, t) - lamb) < 1e-7:
                    right_most = t
            if right_most == None:
                right_most = len(self.knots)

        tmp_knots = [left_new_knot] + self.knots[left_most:right_most] + [right_new_knot]
        tmp_coef_list = [self.get_constant_f(-lamb)] + self.coef_list[left_most:right_most+1] + [self.get_constant_f(lamb)]

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


def solver(y: np.array, lamb: float, loss: str = None) -> np.array:
    n = y.size
    delta_squared = [None] * n
    beta = np.zeros(n)
    delta_squared[0] = DeltaSquared(knots = [], coef_list=[[1, -y[0]]])
    for i in range(n - 1):
        delta_squared[i + 1] = delta_squared[i].forward(
            lamb, y[i + 1])
        print(f"delta_squared[{i + 1}]:", vars(delta_squared[i + 1]))
    beta[n - 1] = delta_squared[n - 1].find_tangency(0)
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta_squared[i-1].backward(next_beta=beta[i])
    return beta


if __name__ == "__main__":
    beta = solver(np.array([0,1]), 0.5)
    print(beta)
