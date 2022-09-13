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
        self.knots_n = len(self.knots)
        self.coef_list = coef_list
        assert len(self.coef_list) == self.knots_n + 1

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
        next_knots, next_f_coef_list = self.solve_next_knots(lamb)
        loss = self.get_loss(next_yi)
        next_coef_list = [
            tuple([c + e for c, e in zip(fc, loss)]) for fc in next_f_coef_list
        ]
        return (next_knots[0], next_knots[-1], next_knots, next_coef_list)

    def solve_next_knots(self, lamb: float):
        """solve the value of bs for next delta's knots. Also calculate ab and ac of the next f function

        Args:
            lamb (float): lambda
            next_bm (_type_):
            next_bp (_type_): _description_

        Returns:
            _type_: _description_
        """

        for i in range(self.knots_n, 0, -1):
            if self.calc_derivative_at(i - 1) < lamb:
                break
        next_bp = self.calc_inverse(i, lamb)
        right_survive_knot_index = i

        for i in range(self.knots_n):
            if self.calc_derivative_at(i) > -lamb:
                break
        next_bm = self.calc_inverse(i, -lamb)
        left_survive_knot_index = i

        survive_knots = self.knots[left_survive_knot_index:right_survive_knot_index]
        survive_coef_list = self.coef_list[
            left_survive_knot_index: right_survive_knot_index + 1
        ]
        next_knots = [next_bm] + survive_knots + [next_bp]
        next_f_coef_list = [(0, -lamb)] + survive_coef_list + [(0, lamb)]
        return next_knots, next_f_coef_list

    def find_min(self):
        for t in range(self.knots_n):
            if self.calc_derivative_at(t) >= 0:
                return self.calc_inverse(t, 0)
        return self.calc_inverse(self.knots_n, 0)

    @abstractmethod
    def get_loss(self, next_yi):
        pass

    @abstractmethod
    def calc_inverse(self, t, d):
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

    def calc_derivative_at(self, t):
        return sum(
            [
                c * bf(self.knots[t])
                for bf, c in zip(self.basis_function_list, self.coef_list[t])
            ]
        )

    def calc_inverse(self, t, d):
        assert self.coef_list[t][0] != 0
        return (d - self.coef_list[t][1]) / self.coef_list[t][0]

    def get_loss(self, next_yi):
        return (1, -next_yi)


def solver(y: np.array, lamb: float, loss: str = None) -> np.array:
    n = y.size
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared(
        bm=None, bp=None, knots=[np.inf], coef_list=[(1, -y[0]), (1, -y[0])]
    )
    for i in range(n - 1):
        bm, bp, knots, coef_list = delta_squared[i].forward(
            lamb, y[i + 1])
        delta_squared[i + 1] = DeltaSquared(
            bm=bm, bp=bp, knots=knots, coef_list=coef_list
        )
        print(f"delta_squared[{i + 1}]:", vars(delta_squared[i + 1]))
    beta[n - 1] = delta_squared[n - 1].find_min()
    print("backward!!")
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta_squared[i].backward(next_beta=beta[i])
    return beta


if __name__ == "__main__":
    beta = solver(np.array([1, 0, 2, 0, 3, 1, 2]), 0.5)
    print(beta)
