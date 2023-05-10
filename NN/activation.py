import numpy as np


class Function:
    def __init__(self):
        self.unactivated_output = None

    def output(self, x):
        raise NotImplementedError

    def unactivated_error(self, layer_output_error):
        raise NotImplementedError


class Tanh(Function):
    def __init__(self):
        super().__init__()

    def output(self, x):
        self.unactivated_output = x
        return np.tanh(x)

    def unactivated_error(self, layer_output_error):
        return layer_output_error * self.func_prime(self.unactivated_output)

    def func_prime(self, x):
        return 1 - self.output(x) ** 2


class Softmax(Function):
    def __init__(self):
        super().__init__()
        self.activated_output = None

    def output(self, x):
        self.unactivated_output = x
        temp = np.exp(x)
        self.activated_output = temp / (np.sum(temp, axis=0))
        return self.activated_output

    def unactivated_error(self, layer_output_error):
        n, m = np.shape(layer_output_error)
        unactivated_error = np.zeros((n, m))
        for i in range(m):
            temp = np.tile(self.activated_output[:, [i]], n)
            unactivated_error[:, [i]] = np.matmul(temp * (np.identity(n) - temp.T), layer_output_error[:, [i]])
        return unactivated_error


class Sigmoid(Function):
    def __init__(self):
        super().__init__()

    def output(self, x):
        self.unactivated_output = x
        return 1 / (1 + np.exp(-x))

    def unactivated_error(self, layer_output_error):
        return layer_output_error * self.func_prime(self.unactivated_output)

    def func_prime(self, x):
        return self.output(x) * (1 - self.output(x))


class ReLU(Function):
    def __init__(self):
        super().__init__()

    def output(self, x):
        self.unactivated_output = x
        return np.maximum(x, 0)

    def unactivated_error(self, layer_output_error):
        return layer_output_error * self.func_prime(self.unactivated_output)

    def func_prime(self, x):
        return x > 0
