from layers import GeneratorLayer, GeneratorOutputLayer, DiscriminatorLayer, DiscriminatorOutputLayer
import matplotlib.pyplot as plt
import numpy as np

from os import makedirs, path
from joblib import dump


class GAN:
    def __init__(self, latent_vector_dim, output_dim, lr=0.001):
        self.input_dim = latent_vector_dim  # dimensionality of the latent vector Z_rand
        self.output_dim = output_dim    # dimensionality of an MNIST image is 28*28 = 784
        self.learning_rate = lr

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


    def generator_GD_update_step(self):
        for layer in self.generator_layers:
            layer.W -= self.learning_rate * layer.dL_dW
            layer.B -= self.learning_rate * layer.dL_dB


    def discriminator_GD_update_step(self):
        for layer in self.discriminator_layers:
            layer.W -= self.learning_rate * layer.dL_dW
            layer.B -= self.learning_rate * layer.dL_dB
    

    def infer(self, num_samples):
        """
        Generates random noise, passes it through the Generator and displays the image.
        """
        # Generate Noise Vector z
        z_noise = np.random.randn(num_samples, self.input_dim)
        
        # Forward Pass through Generator
        generated_images = self.gen_forward(z_noise)
        
        side_len = int(np.sqrt(self.output_dim))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1: axes = [axes] # Handle single image case
            
        for i, ax in enumerate(axes):
            img = generated_images[i].reshape(side_len, side_len)
            
            ax.imshow(img, cmap='Greys_r')
            ax.axis('off')
            ax.set_title(f"Gen {i+1}")
        
        plt.show()
    

    def save_model(self, filename):
        """
        Saves the model weights to a file.
        """
        directory = path.dirname(filename)
        if directory:
            makedirs(directory, exist_ok=True)

        model_data = {
            # Generator Weights
            "gen_layers": [
                {"W": layer.W, "B": layer.B} for layer in self.generator_layers
            ],
            # Discriminator Weights
            "disc_layers": [
                {"W": layer.W, "B": layer.B} for layer in self.discriminator_layers
            ],
            # Hyperparameters
            "config": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "lr": self.learning_rate
            }
        }

        dump(model_data, filename)
        print(f"GAN model successfully saved to {filename}")