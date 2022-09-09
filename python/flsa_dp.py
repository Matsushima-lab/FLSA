from __future__ import annotations
from colorsys import yiq_to_rgb

from turtle import forward


class DeltaFunc:
    def __init__(self, lamb, knots) -> None:
        self.lamb = lamb
        self.knots = knots
        self.knots_len = len(self.knots)

    # TODO -> takeda kun
    def backward(self, next_beta):
        """
        Args:
        Return:beta
        """
        bm, bp = self.new_knots()
        return max(min(next_beta, bp), bm)
        """
        self.b is a float or a set of float that satisfies delta' = +-lambda
        Calculated in the process of "forward" method
        """

    def forward(self, yi: float) -> DeltaFunc:
        """
        Compute next delta(b) as min_b' delta(b') + loss(b,yi) +  lambda |b'-b|

        Args:
        Return; DeltaFunc
        """
        return DeltaFunc()

    def find_min(self):
        """ """
        return 0


class Deltalogistic(DeltaFunc):
    def forward(self, lamb, yi) -> DeltaFunc:
        pass


class DeltaSquared(DeltaFunc):
    def __init__(
        self,
        lamb,
        knots,
        slopes,
        intercepts,
    ):
        """init function for
            delta'(b)のi番目のknot区間について
            delta'(b) = a_b_list[i] * b + a_c_list[i]

        Args:
            knot (_type_): _description_
            a_b_list (_type_): _description_
            a_c_list (_type_): _description_
        """

        super().__init__(lamb, knots)
        self.slopes = slopes #list of slopes
        self.intercepts = intercepts #list of intercepts

    def forward(self, next_yi: float) -> DeltaSquared:
        """

        Args:
            lamb (float): hiperparameter lambda
            next_yi (float): y[i+1]

        Returns:
            DeltaSquared: next delta function
        """
        #calculate f'(b)
        next_knots, survive_slopes, survive_intercepts = self.solve_next_knots()

        slope_increment = 1
        intercept_increment = -next_yi

        #calculate e'(b) + f'(b) (=delta'(b))
        next_slopes = [slope + slope_increment for slope in survive_slopes]
        next_intercepts = [intercept + intercept_increment for intercept in survive_intercepts]


        return DeltaSquared(self.lamb, next_knots, next_slopes, next_intercepts)

    def find_min(self):
        for i in range(self.knots_len):
            if (
                self.calc_y_from_b(
                    self.knots[i], self.slopes[i], self.intercepts[i]
                )
                >= 0
            ):

                return self.calc_b_from_y(0, self.slopes[i], self.intercepts[i])

        return self.calc_b_from_y(0, self.slopes[self.knots_len], self.intercepts[self.knots_len])

    def solve_next_knots(self):
        """solve the value of bs for next delta's knots. Also calculate ab and ac of the next f function

        Args:

        Returns:
            _type_: _description_
        """
        if self.knots_len > 0:
            none_survived = False

            reached_mlamb = False
            for i in range(self.knots_len):
                if (
                    self.calc_y_from_b(
                        self.knots[i], self.slopes[i], self.intercepts[i]
                    )
                    > -self.lamb
                ):
                    left_survive_knot_index = i
                    new_knot_line_m = (self.slopes[i], self.intercepts[i])
                    reached_mlamb = True
                    break
            #if all knots are under -lamb
            if (i == self.knots_len-1 and reached_mlamb == False): none_survived = True

            reached_plamb = False
            for i in range(self.knots_len, 0, -1):
                if (
                    self.calc_y_from_b(
                        self.knots[i - 1], self.slopes[i], self.intercepts[i]
                    )
                    < self.lamb
                ):
                    right_survive_knot_index = i - 1
                    new_knot_line_p = (self.slopes[i], self.intercepts[i])
                    reached_plamb = True
                    break
            #if all knots are beyond +lamb
            if (i == 1 and reached_plamb == False): none_survived = True


            if (none_survived == False):
                survive_knots = self.knots[
                    left_survive_knot_index: right_survive_knot_index + 1
                ]
                survive_slopes = self.slopes[
                    left_survive_knot_index: right_survive_knot_index + 2
                ]
                survive_intercepts = self.intercepts[
                    left_survive_knot_index: right_survive_knot_index + 2
                ]

            else:
                if (reached_mlamb): 
                    new_knot_line_m, new_knot_line_p, survive_knots, survive_slopes, survive_intercepts = self.start_from_empty_knots(0)
                else: 
                    new_knot_line_m, new_knot_line_p, survive_knots, survive_slopes, survive_intercepts = self.start_from_empty_knots(self.knots_len)

        else:
            new_knot_line_m, new_knot_line_p, survive_knots, survive_slopes, survive_intercepts = self.start_from_empty_knots(0)

        self.left_new_knot = self.calc_b_from_y(
            -self.lamb, new_knot_line_m[0], new_knot_line_m[1]
        )
        self.right_new_knot = self.calc_b_from_y(
            self.lamb, new_knot_line_p[0], new_knot_line_p[1]
        )
        next_knots = [self.left_new_knot] + survive_knots + [self.right_new_knot]
        survive_slopes = [0] + survive_slopes + [0]
        survive_intercepts = [-self.lamb] + survive_intercepts + [self.lamb]
        return next_knots, survive_slopes, survive_intercepts

    def calc_y_from_b(self, b, slope, intercept):
        return b * slope + intercept

    def calc_b_from_y(self, y, slope, intercept):
        assert slope != 0
        return (y - intercept) / slope

    def new_knots(self):
        return (self.left_new_knot, self.right_new_knot)

    def start_from_empty_knots(self,i):
        new_knot_line_m = (self.slopes[i], self.intercepts[i])
        new_knot_line_p = (self.slopes[i], self.intercepts[i])
        survive_knots = []
        survive_slopes = [self.slopes[i]]
        survive_intercepts = [self.intercepts[i]]
        return new_knot_line_m, new_knot_line_p, survive_knots, survive_slopes, survive_intercepts

def main(y, lamb: float, loss: str = None):
    n = len(y)
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared(
        lamb = lamb, knots=[], slopes=[1], intercepts=[-y[0]]
    )
    for i in range(n - 1):
        delta_squared[i + 1] = delta_squared[i].forward(y[i + 1])
        print(f'delta_squared[{i + 1}]:', vars(delta_squared[i+1]))
    beta[n - 1] = delta_squared[n - 1].find_min()
    print("backward!!")
    for i in range(n - 1, 0, -1):
        beta[i - 1] = delta_squared[i-1].backward(next_beta=beta[i])
    return beta


if __name__ == "__main__":
    beta = main([0,1], 0.5)
    print(beta)
