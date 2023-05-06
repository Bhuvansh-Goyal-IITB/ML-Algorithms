import numpy as np
from layer import Layer
from activation import Function


class Dense(Layer):
    def __init__(self, input_shape, output_shape, activation: Function = None):
        super().__init__()
        self.activation = activation
        self.weight = np.random.rand(output_shape, input_shape) - 0.5
        self.bias = np.random.rand(output_shape, 1) - 0.5

    def forward_propagation(self, layer_input):
        self.input = layer_input
        self.output = np.matmul(self.weight, self.input) + self.bias
        if self.activation is not None:
            return self.activation.output(self.output)
        else:
            return self.output

    def back_propagation(self, layer_output_error, learning_rate):
        if self.activation is not None:
            unactivated_error = self.activation.unactivated_output(layer_output_error)
        else:
            unactivated_error = layer_output_error

        weight_error = np.matmul(unactivated_error, np.transpose(self.input))
        bias_error = np.reshape(np.sum(unactivated_error, axis=1), np.shape(self.bias))

        self.weight -= learning_rate * weight_error
        self.bias -= learning_rate * bias_error

        return np.matmul(np.transpose(self.weight), unactivated_error)
