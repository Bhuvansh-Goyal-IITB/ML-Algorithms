import numpy
import numpy as np


class LinearRegression:
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray):
        self.X = x
        self.Y = y

        self.m = x.shape[1]
        self.w = np.zeros((1, x.shape[0]))
        self.b = 0
        self.model_ready = False

    def predict(self):
        if not self.model_ready:
            print("Please train the model first")
            return

        return np.dot(self.w, self.X) + self.b

    def fit_itr(self, iterations, learning_rate):
        w = 0
        b = 0

        for i in range(iterations):
            gradient_w = 0
            gradient_b = 0
            for j in range(self.m):
                gradient_w += (w * self.X[0, j] + b - self.Y[0, j]) * self.X[0, j]
                gradient_b += (w * self.X[0, j] + b - self.Y[0, j])

            gradient_w /= self.m
            gradient_b /= self.m

            if i % 100 == 0:
                print(gradient_w)

            w -= learning_rate * gradient_w
            b -= learning_rate * gradient_b

        self.model_ready = True
        print("Model trained")
        return w, b

    def fit(self, iterations, learning_rate):

        self.w = np.zeros((1, self.X.shape[0]))
        self.b = 0

        for _ in range(iterations):
            A = np.dot(self.w, self.X) + self.b
            Z = A - self.Y

            gradient_w = (1 / self.m) * self.X.dot(Z.T)

            if _ % 100 == 0:
                print(gradient_w)

            self.w = self.w - (learning_rate * gradient_w.T)
            self.b -= ((learning_rate / self.m) * np.sum(Z))

        self.model_ready = True
        print("Model trained")
        return self.w, self.b
