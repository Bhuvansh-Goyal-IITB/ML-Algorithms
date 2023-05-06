import numpy as np


class Error:
    def error(self, y_calc, y_exp):
        raise NotImplementedError

    def error_prime(self, y_calc, y_exp):
        raise NotImplementedError


class MSE(Error):
    def error(self, y_calc, y_exp):
        return np.mean(np.power(y_exp - y_calc, 2))

    def error_prime(self, y_calc, y_exp):
        return (2 / len(y_calc)) * (y_calc - y_exp)
