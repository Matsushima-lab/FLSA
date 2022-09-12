from __future__ import annotations
from abc import abstractmethod
from colorsys import yiq_to_rgb
import numpy as np

from turtle import forward


class DeltaFunc:

    basis_function_list = []

    def __init__(self, bm, bp, knots, coef_list) -> None:
        self.bm = bm
        self.bp = bp
        self.knots = knots
        self.knots_n = len(self.knots)
        self.coef_list = coef_list

    # TODO -> takeda kun
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

    def forward(self, lamb: float, yi: float, next_yi: float) -> DeltaFunc:
        """
        Compute next delta(b) as min_b' delta(b') + loss(b,yi) +  lambda |b'-b|

        Args:
        Return; DeltaFunc
        """
        next_knots, next_f_coef_list = self.solve_next_knots(
            lamb
        )
        print(next_f_coef_list)
        e_coef = self.get_e_coef(next_yi)
        next_coef_list = [tuple([c + e for c, e in zip(fc, e_coef)])
                          for fc in next_f_coef_list]

        return DeltaSquared(bm=next_knots[0], bp=next_knots[-1], knots=next_knots, coef_list=next_coef_list)

    def solve_next_knots(self, lamb: float):
        """solve the value of bs for next delta's knots. Also calculate ab and ac of the next f function

        Args:
            lamb (float): lambda
            next_bm (_type_):
            next_bp (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(self.knots) > 0:
            for i in range(self.knots_n):
                if (
                    self.calc_y_from_b(
                        self.knots[i], self.coef_list[i]
                    )
                    > -lamb
                ):
                    left_survive_knot_index = i
                    new_knot_line_m = self.coef_list[i]
                    break
            for i in range(self.knots_n, 0, -1):
                if (
                    self.calc_y_from_b(
                        self.knots[i - 1], self.coef_list[i]
                    )
                    < lamb
                ):
                    right_survive_knot_index = i - 1
                    new_knot_line_p = self.coef_list[i]
                    break
            survive_knots = self.knots[
                left_survive_knot_index: right_survive_knot_index + 1
            ]
            survive_coef_list = self.coef_list[
                left_survive_knot_index: right_survive_knot_index + 2
            ]

        else:
            new_knot_line_p = new_knot_line_m = self.coef_list[0]
            survive_knots = []
            survive_coef_list = [self.coef_list[0]]

        next_bm = self.calc_b_from_y(
            -lamb, new_knot_line_m
        )
        next_bp = self.calc_b_from_y(
            lamb, new_knot_line_p
        )
        next_knots = [next_bm] + survive_knots + [next_bp]
        next_f_coef_list = [(0, -lamb)] + survive_coef_list + [(0, lamb)]
        return next_knots, next_f_coef_list

    def find_min(self):
        for i in range(self.knots_n):
            if (
                self.calc_y_from_b(
                    self.knots[i], self.coef_list[i]
                )
                >= 0
            ):
                return self.calc_b_from_y(0, self.coef_list[i])

        return self.calc_b_from_y(0, self.coef_list[self.knots_n])

    @abstractmethod
    def get_e_coef(self, next_yi):
        pass

    @abstractmethod
    def calc_b_from_y(self, y, coef):
        pass

    @abstractmethod
    def calc_y_from_b(self, b, coef):
        pass


class Deltalogistic(DeltaFunc):
    def forward(self, lamb, yi) -> DeltaFunc:
        pass


class DeltaSquared(DeltaFunc):
    basis_function_list = [lambda x:x, lambda x:1]

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
            knot (_type_): _description_
            a_b_list (_type_): _description_
            a_c_list (_type_): _description_
        """

        super().__init__(bm=bm, bp=bp, knots=knots, coef_list=coef_list)

    def calc_y_from_b(self, b, coef):
        return sum([c * bf(b) for bf, c in zip(self.basis_function_list, coef)])

    def calc_b_from_y(self, y, coef):
        assert coef[0] != 0
        return (y - coef[1]) / coef[0]

    def get_e_coef(self, next_yi):
        return (1, -next_yi)


def main(y: np.array, lamb: float, loss: str = None) -> np.array:
    n = y.size
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared(
        bm=None, bp=None, knots=[], coef_list=[(1, -y[0])]
    )
    for i in range(n - 1):
        delta_squared[i + 1] = delta_squared[i].forward(lamb, y[i], y[i + 1])
        print(f'delta_squared[{i + 1}]:', vars(delta_squared[i+1]))
    beta[n - 1] = delta_squared[n - 1].find_min()
    print("backward!!")
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta_squared[i].backward(next_beta=beta[i])
    return beta


if __name__ == "__main__":
    beta = main(np.array([1, 0]), 0.5)
    print(beta)
