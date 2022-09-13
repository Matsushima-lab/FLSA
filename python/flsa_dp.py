from __future__ import annotations
from abc import abstractmethod
from colorsys import yiq_to_rgb
from typing import overload
import numpy as np
from abc import ABC, abstractmethod

from turtle import forward


class DeltaFunc(ABC):

    basis_function_list = []

    def __init__(self, bm, bp, knots, coef_list) -> None:
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
        self.bm = bm
        self.bp = bp
        self.knots = knots
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
        next_delta = self.add_loss(next_f)
        return next_delta

    @abstractmethod
    def find_tangency(self, g):
        pass

    @abstractmethod
    def overwrite(self, left_new_knot, right_new_knot):
        pass

    @abstractmethod
    def add_loss(self, next_yi):
        pass

    @abstractmethod
    def get_constant_f(self, x):
        pass

    @abstractmethod
    def calc_inverse_spline(self, t, d):
        pass

    @abstractmethod
    def calc_derivative_at(self, t):
        pass


class Deltalogistic(DeltaFunc):
    pass


class DeltaSquared(DeltaFunc):
    basis_function_list = [lambda x: x, lambda x: 1]

    def __init__(
        self,
        bm,
        bp,
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
        super().__init__(bm=bm, bp=bp, knots=knots, coef_list=coef_list)

    def get_constant_f(self, x):
        return (0, x)

    def calc_derivative_at(self, t):
        return sum(
            [
                c * bf(self.knots[t])
                for bf, c in zip(self.basis_function_list, self.coef_list[t])
            ]
        )

    def calc_inverse_spline(self, t, d):
        assert self.coef_list[t][0] != 0
        return (d - self.coef_list[t][1]) / self.coef_list[t][0]

    def calc_eprime(self, next_yi):
        return (1, -next_yi)


def solver(y: np.array, lamb: float, loss: str = None) -> np.array:
    delta_squared[0] = DeltaSquared()
    for i in range(n - 1):
        delta_squared[i + 1] = delta_squared[i].forward(
            lamb, y[i + 1])
        print(f"delta_squared[{i + 1}]:", vars(delta_squared[i + 1]))
    beta[n - 1] = delta_squared[n - 1].find_tangency(0)
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta_squared[i].backward(next_beta=beta[i])
    return beta


if __name__ == "__main__":
    beta = solver(np.array([0,1]), 0.5)
    print(beta)
