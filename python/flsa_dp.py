

class DeltaFunc():
    def __init__(self) -> None:
        pass

    def backward(self, beta):
        '''
        Args:
        Return:beta
        '''
        pass

    def forward(lamb, prev_delta, loss):
        '''
        Args:
        Return; DeltaFunc
        '''
        return DeltaFunc()

    def find_min(self):
        return 0


class Deltalogistic(DeltaFunc):
    pass


class DeltaSquared(DeltaFunc):
    pass


if __name__ == "__main__":
    n = 10
    delta_squared = [None] * n
    beta = [0] * n
    delta_squared[0] = DeltaSquared()
    for i in range(n):
        delta_squared[i+1] = delta_squared[i-1].forward()
    beta[n-1] = delta_squared[n-1].find_min()
    for i in range(n-1, 0):
        beta[i] = delta_squared[i].backward()
