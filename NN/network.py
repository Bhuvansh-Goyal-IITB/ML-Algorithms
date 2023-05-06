from NN.error import Error
from NN.layer import Layer


class Network:
    def __init__(self, error: Error = None):
        self.layers: [Layer] = []
        self.error = error

    def add(self, *layers):
        for layer in layers:
            self.layers.append(layer)

    def set_error_func(self, error: Error):
        self.error = error

    def fit(self, input_data, output_data, epochs, learning_rate):
        for layer in self.layers:
            layer.reset_parameters()

        if self.error is None:
            print("No error function assigned")
            return

        for i in range(epochs):
            output = self.predict(input_data)
            error = self.error.error_prime(output, output_data)
            total_error = self.error.error(output, output_data)
            for layer in self.layers[::-1]:
                error = layer.back_propagation(error, learning_rate)

            print(f"Epoch: {i+1}, Error: {total_error}")

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output
