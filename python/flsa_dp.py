

from turtle import forward


class DeltaFunc():
    def __init__(self) -> None:
        pass

# TODO -> takeda kun
    def backward(self, beta):
        '''
        Args:
        Return:beta
        '''
        pass

    def forward(self, lamb: float, yi: float) -> DeltaFunc:
        '''
        Compute next delta(b) as min_b' delta(b') + loss(b,yi) +  lambda |b'-b|

        Args:
        Return; DeltaFunc
        '''
        return DeltaFunc()

    def find_min(self):
        '''
        '''
        return 0


class Deltalogistic(DeltaFunc):
    def forward(self, lamb, yi) -> Deltalogistic:
        pass


class DeltaSquared(DeltaFunc):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def forward(self, lamb: float, yi: float) -> DeltaFunc:
        pass


def main(y: np.array, lamb: float, loss: str) -> np.array:
    n = y.size
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared()
    for i in range(n):
        delta_squared[i+1] = delta_squared[i-1].forward()
    beta[n-1] = delta_squared[n-1].find_min()
    for i in range(n-1, 0):
        beta[i] = delta_squared[i].backward()
    return beta


if __name__ == "__main__":
    main()
