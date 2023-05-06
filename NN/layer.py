class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.activation = None

    def forward_propagation(self, layer_input):
        raise NotImplementedError

    def back_propagation(self, layer_output_error, learning_rate):
        raise NotImplementedError
