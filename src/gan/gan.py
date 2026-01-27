from layers import GeneratorLayer, GeneratorOutputLayer, DiscriminatorLayer, DiscriminatorOutputLayer

class GAN:
    def __init__(self, latent_vector_dim, output_dim, learning_rate=0.001):
        self.input_dim = latent_vector_dim  # dimensionality of the latent vector Z_rand
        self.output_dim = output_dim    # dimensionality of an MNIST image is 28*28 = 784
        self.learning_rate = learning_rate

        self.generator_layers = [
            GeneratorLayer(self.input_dim, 256),
            GeneratorLayer(256, 512),
            GeneratorOutputLayer(512, self.output_dim)
        ]

        self.discriminator_layers = [
            DiscriminatorLayer(self.output_dim, 512),
            DiscriminatorLayer(512, 256),
            DiscriminatorOutputLayer(256, 1)
        ]

    
    def gen_forward(self, Z):
        A = Z
        for layer in self.generator_layers:
            A = layer.forward(A)
        
        gen_output = A
        return gen_output
    

    def disc_forward(self, X):
        A = X
        for layer in self.discriminator_layers:
            A = layer.forward(A)

        disc_output = A
        return disc_output


    def disc_backward(self, y_target):
        gradient = self.discriminator_layers[-1].backward(y_target)

        for layer in self.discriminator_layers[-2::-1]:
            gradient = layer.backward(gradient)
        
        return gradient


    def gen_backward(self, gradient_from_discriminator):
        gradient = gradient_from_discriminator

        for layer in self.generator_layers[::-1]:
            gradient = layer.backward(gradient)
        
        return gradient
    

    def reset_gradients(self):
        for layer in (self.generator_layers + self.discriminator_layers):
            layer.set_gradients_zero()
    

    def GD_update_step(self):
        for layer in (self.generator_layers + self.discriminator_layers):
            layer.W -= self.learning_rate * layer.dL_dW
            layer.B -= self.learning_rate * layer.dL_dB
