import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim):
        # Initialise everything except the Weight matrix because there's a different scale factor in each class
        self.B = np.zeros((1, output_dim))
        
        self.X_input = None

        # Initialise gradients with Zeros (so we can add to them later during backpropagation)
        self.dL_dW = np.zeros((input_dim, output_dim))
        self.dL_dB = np.zeros((1, output_dim))
    

    def forward(self):
        return "Forward Pass"
    

    def backward(self):
        return "Backward Pass"


    def _relu(self, matrix):
        return np.maximum(0, matrix)
    

    def _relu_derivative(self, x):
        # Derivative of _ReLU: 1 if x > 0, else 0
        return (x > 0).astype(float)


    def _sigmoid(self, matrix):
        return 1 / (1 + np.exp(-matrix))

    
    def _sigmoid_derivative(self, matrix):
        return self._sigmoid(matrix) * (1 - self._sigmoid(matrix))


    def reset_gradients(self):
        # We need to empty the bucket before a new epoch
        self.dL_dW.fill(0)
        self.dL_dB.fill(0)
    

class DiscriminatorOutputLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        
        scale_factor = np.sqrt(1.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * scale_factor


    def forward(self, previous_layer_A):
        self.X_input = previous_layer_A # input to the output layer is the post-activation output from the previous layer
        self.Z = np.dot(self.X_input, self.W) + self.B
        self.A = self._sigmoid(self.Z)

        return self.A # this A will be the prediction


    def backward(self, y_target):
        # If Real: (A - 1)
        # If Fake: (A - 0)
        error = self.A - y_target

        self.dL_dW += np.dot(self.X_input.T, error)
        self.dL_dB += np.sum(error, axis=0, keepdims=True)

        gradient_to_pass_back = np.dot(error, self.W.T)

        return gradient_to_pass_back


class DiscriminatorLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        scale_factor = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * scale_factor


    def forward(self, previous_layer_A):
        self.X_input = previous_layer_A
        self.Z = np.dot(self.X_input, self.W) + self.B
        self.A = self._relu(self.Z)
        
        return self.A


    def backward(self, gradient_flowing_back):
        error = gradient_flowing_back * self._relu_derivative(self.Z)
        
        self.dL_dW += np.dot(self.X_input.T, error)
        self.dL_dB += np.sum(error, axis=0, keepdims=True)

        gradient_to_pass_back = np.dot(error, self.W.T)

        return gradient_to_pass_back


class GeneratorOutputLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        
        scale_factor = np.sqrt(1.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * scale_factor


    def forward(self, previous_layer_A):
        self.X_input = previous_layer_A
        self.Z = np.dot(self.X_input, self.W) + self.B
        self.A = self._sigmoid(self.Z)
        
        return self.A


    def backward(self, gradient_flowing_back_from_discriminator):
        error = gradient_flowing_back_from_discriminator * self._sigmoid_derivative(self.Z) 

        self.dL_dW += np.dot(self.X_input.T, error)
        self.dL_dB += np.sum(error, axis=0, keepdims=True)

        gradient_to_pass_back = np.dot(error, self.W.T)

        return gradient_to_pass_back


class GeneratorLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        scale_factor = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * scale_factor


    def forward(self, previous_layer_A):
        self.X_input = previous_layer_A
        self.Z = np.dot(self.X_input, self.W) + self.B
        self.A = self._relu(self.Z)
        
        return self.A


    def backward(self, gradient_flowing_back):
        error = gradient_flowing_back * self._relu_derivative(self.Z)
        
        self.dL_dW += np.dot(self.X_input.T, error)
        self.dL_dB += np.sum(error, axis=0, keepdims=True)

        gradient_to_pass_back = np.dot(error, self.W.T)

        return gradient_to_pass_back

