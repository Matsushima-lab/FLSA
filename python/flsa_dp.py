from __future__ import annotations
from abc import abstractmethod
from colorsys import yiq_to_rgb
import numpy as np

from turtle import forward


class DeltaFunc:

    basis_function_list = []

    def __init__(self, bm, bp, knots) -> None:
        self.bm = bm
        self.bp = bp
        self.knots = knots
        self.knots_n = len(self.knots)

    def forward(self, lamb: float, yi: float, next_yi: float) -> DeltaSquared:
        """

        Args:
            lamb (float): hiperparameter lambda
            yi (float): y[i]
            next_yi (float): y[i+1]

        Returns:
            DeltaSquared: next delta function
        """
        next_bm = yi - lamb
        next_bp = yi + lamb
        next_knots, next_f_coef_list = self.solve_next_knots(
            lamb, next_bm, next_bp
        )
        next_e_ab = 1
        next_e_ac = -next_yi
        next_coef_list = [(c for c in fc) for fc in next_f_coef_list]
        return DeltaSquared(next_knots, next_a_b_list, next_a_c_list, next_bm, next_bp)

    def solve_next_knots(self, lamb: float, next_bm, next_bp):
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
                        self.knots[i], self.a_b_list[i], self.a_c_list[i]
                    )
                    > next_bm
                ):
                    left_survive_knot_index = i
                    new_knot_line_m = (self.a_b_list[i], self.a_c_list[i])
                    break
            for i in range(self.knots_n, 0, -1):
                if (
                    self.calc_y_from_b(
                        self.knots[i - 1], self.a_b_list[i], self.a_c_list[i]
                    )
                    < next_bp
                ):
                    right_survive_knot_index = i - 1
                    new_knot_line_p = (self.a_b_list[i], self.a_c_list[i])
                    break
            survive_knots = self.knots[
                left_survive_knot_index: right_survive_knot_index + 1
            ]
            survive_a_b_list = self.a_b_list[
                left_survive_knot_index: right_survive_knot_index + 2
            ]
            survive_a_c_list = self.a_c_list[
                left_survive_knot_index: right_survive_knot_index + 2
            ]

        else:
            new_knot_line_m = (self.a_b_list[0], self.a_c_list[0])
            new_knot_line_p = (self.a_b_list[0], self.a_c_list[0])
            survive_knots = []
            survive_a_b_list = [self.a_b_list[0]]
            survive_a_c_list = [self.a_c_list[0]]

        left_new_knot = self.calc_b_from_y(
            next_bm, new_knot_line_m[0], new_knot_line_m[1]
        )
        right_new_knot = self.calc_b_from_y(
            next_bp, new_knot_line_p[0], new_knot_line_p[1]
        )
        next_knots = [left_new_knot] + survive_knots + [right_new_knot]
        next_f_a_b_list = [0] + survive_a_b_list + [0]
        next_f_a_c_list = [-lamb] + survive_a_c_list + [lamb]
        return next_knots, next_f_a_b_list, next_f_a_c_list

    def find_min(self):
        for i in range(self.knots_n):
            if (
                self.calc_y_from_b(
                    self.knots[i], self.a_b_list[i], self.a_c_list[i]
                )
                >= 0
            ):

                return self.calc_b_from_y(0, self.a_b_list[i], self.a_c_list[i])

        return self.calc_b_from_y(0, self.a_b_list[self.knots_n], self.a_c_list[self.knots_n])

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

    def forward(self, lamb: float, yi: float) -> DeltaFunc:
        """
        Compute next delta(b) as min_b' delta(b') + loss(b,yi) +  lambda |b'-b|

        Args:
        Return; DeltaFunc
        """
        return DeltaFunc()

    def find_min(self):
        """ """
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
    def calc_b_from_y(self, y, coef):
        pass

    @abstractmethod
    def calc_y_from_b(self, b, coef):
        pass


class Deltalogistic(DeltaFunc):
    def forward(self, lamb, yi) -> DeltaFunc:
        pass


class DeltaSquared(DeltaFunc):
    basis_function_list = [lambda x:x, 1]

    def __init__(
        self,
        knots,
        coef_list,
        bm,
        bp,
    ):
        """init function for
            delta'(b)のi番目のknot区間について
            delta'(b) = a_b_list[i] * b + a_c_list[i]

        Args:
            knot (_type_): _description_
            a_b_list (_type_): _description_
            a_c_list (_type_): _description_
        """

        super().__init__(bm, bp, knots)
        self.coef_list = coef_list

    def calc_y_from_b(self, b, ab, ac):
        return b * ab + ac

    def calc_b_from_y(self, y, ab, ac):
        assert ab != 0
        return (y - ac) / ab


def main(y: np.array, lamb: float, loss: str = None) -> np.array:
    n = y.size
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared(
        knots=[], bm=None, bp=None, a_b_list=[1], a_c_list=[0]
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
    beta = main(np.array([0, 1]), 0.5)
    print(beta)
