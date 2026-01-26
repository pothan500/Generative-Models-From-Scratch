"""
Encoder_in (784) -> Encoder_hl_1 (512) -> Latent space (2/3) -> Decoder_hl_1 (512) -> Decoder_out (784)
"""

from os import makedirs, path
import numpy as np
import matplotlib.pyplot as plt
from joblib import load, dump

class VAE:
    def __init__(self, n, hidden_layer_dim, k_latent, learning_rate=0.001):
        self.n = n # 784 for MNIST
        self.hidden_layer_dim = hidden_layer_dim
        self.k_latent = k_latent
        self.learning_rate = learning_rate

        # ADAM Hyperparamters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_epsilon = 1e-8

        # Initialise Weights (W), Biases (B), Momentum (v) and RMSProp (s) matrices
        scale_input_weights = np.sqrt(2.0 / n)
        scale_hl_weights = np.sqrt(2.0 / hidden_layer_dim)
        scale_latent_input = np.sqrt(2.0 / k_latent)

        (self.enc_W_1, self.enc_B_1, self.v_enc_W_1, self.v_enc_B_1, self.s_enc_W_1, self.s_enc_B_1) = self._init_layer(n, hidden_layer_dim, scale_input_weights)
        (self.W_mu, self.B_mu, self.v_W_mu, self.v_B_mu, self.s_W_mu, self.s_B_mu) = self._init_layer(hidden_layer_dim, k_latent, scale_hl_weights)
        (self.W_sigma, self.B_sigma, self.v_W_sigma, self.v_B_sigma, self.s_W_sigma, self.s_B_sigma) = self._init_layer(hidden_layer_dim, k_latent, scale_hl_weights)
        (self.dec_W_1, self.dec_B_1, self.v_dec_W_1, self.v_dec_B_1, self.s_dec_W_1, self.s_dec_B_1) = self._init_layer(k_latent, hidden_layer_dim, scale_latent_input)
        (self.dec_W_L, self.dec_B_L, self.v_dec_W_L, self.v_dec_B_L, self.s_dec_W_L, self.s_dec_B_L) = self._init_layer(hidden_layer_dim, n, scale_hl_weights)
    

    def _init_layer(self, k_previous, k_to, scale_factor):
        """
        :param k_previous: number of neurons in the layer it connects FROM
        :param k_to: number of neurons in the layer it connects TO
        :param scale_factor: Description
        """

        W = np.random.randn(k_previous, k_to) * scale_factor
        B = np.zeros((1, k_to))

        v_W = np.zeros_like(W)
        s_W = np.zeros_like(W)

        v_B = np.zeros_like(B)
        s_B = np.zeros_like(B)

        return W, B, v_W, v_B, s_W, s_B


    def _relu(self, matrix):
        return np.maximum(0, matrix)

    def _sigmoid(self, matrix):
        return 1 / (1 + np.exp(-matrix))


    def forward_pass(self, X):
        """
        Performs one forward pass.
        
        :param X: Input matrix of shape (Batch_Size, 784) = (m, n)
        """
        self.X = X
        # Encoder
        self.enc_Z_1 = np.dot(self.X, self.enc_W_1) + self.enc_B_1 # (m, n) x (n, k_1) + (m, k_1) = (m, k_1)
        self.enc_A_1 = self._relu(self.enc_Z_1) # (m, k_1)

        enc_Z_mu = np.dot(self.enc_A_1, self.W_mu) + self.B_mu # (m, k_l-1) x (k_l-1, k_latent) + (m, k_latent)= (m, k_latent)
        self.mu = enc_Z_mu
        enc_Z_sigma = np.dot(self.enc_A_1, self.W_sigma) + self.B_sigma # (m, k_l-1) x (k_l-1, k_latent) + (m, k_latent)= (m, k_latent)
        self.sigma = np.exp(enc_Z_sigma) # (m, k_latent)

        # Latent Space
        m = X.shape[0] # batch size
        self.epsilon = np.random.randn(m, self.k_latent)
        self.Z_latent = self.mu + (self.sigma * self.epsilon) # ((m, k_latent) * (m, k_latent)) + (m, k_latent) = (m, k_latent)

        # Decoder
        self.X_dec = self.Z_latent # (m, k_latent)
        self.dec_Z_1 = np.dot(self.X_dec, self.dec_W_1) + self.dec_B_1 # (m, k_latent) x (k_latent, k_1) + (m, k_1) = (m, k_1)
        self.dec_A_1 = self._relu(self.dec_Z_1) # (m, k_1)

        self.dec_Z_L = np.dot(self.dec_A_1, self.dec_W_L) + self.dec_B_L # (m, k_1) x (k_1, n) + (m, n) = (m, n)
        self.dec_A_L = self._sigmoid(self.dec_Z_L) # (m, n)
        
        self.X_hat = self.dec_A_L # (m, n)

        return self.X_hat, self.mu, self.sigma
    

    def _relu_derivative(self, x):
        # Derivative of _ReLU: 1 if x > 0, else 0
        return (x > 0).astype(float)


    def _sigmoid_derivative(self, matrix):
        return self._sigmoid(matrix) * (1 - self._sigmoid(matrix))
    

    def backward_pass(self, alpha_hyper_param):
        """
        Performs one backward pass.
        
        :param alpha_hyper_param: Hyperparameter in the loss function to balance the MSE term and the KL divergence term
        """
        m = self.X.shape[0] # batch size

        # calculate the reconstruction error
        self.delta_recon_output = (-2/m) * (self.X - self.X_hat) * self._sigmoid_derivative(self.dec_Z_L)

        # calculate the reconstruction error at the decoder output layer
        self.delta_recon_dec_W_L = np.dot(self.dec_A_1.T, self.delta_recon_output)

        # calculate the reconstruction error arriving at the decoder
        self.delta_decoder = np.dot(self.delta_recon_output, self.dec_W_L.T) * self._relu_derivative(self.dec_Z_1)

        # calculate the reconstruction error at the decoder first layer
        self.delta_recon_dec_W_1 = np.dot(self.X_dec.T, self.delta_decoder)
        
        # calculate the reconstruction error arriving at the latent space for W_mu
        self.delta_recon_latent_W_mu = np.dot(self.delta_decoder, self.dec_W_1.T)
        # calculate reconstruction error at W_mu
        self.delta_recon_W_mu = np.dot(self.enc_A_1.T, self.delta_recon_latent_W_mu)
        # calculate the KL error for the mean
        self.delta_KL_mu_output = (2/m) * self.mu
        # calculate KL error at W_mu
        self.delta_KL_W_mu = np.dot(self.enc_A_1.T, self.delta_KL_mu_output)

        # calculate reconstruction arriving at the latent space for W_sigma
        self.delta_recon_latent_W_sigma = np.dot(self.delta_decoder, self.dec_W_1.T) * self.epsilon
        # calculate reconstruction error at W_sigma
        self.delta_recon_W_sigma = np.dot(self.enc_A_1.T, (self.delta_recon_latent_W_sigma * self.sigma)) 
        # calculate the KL error for the standard deviation
        self.delta_KL_sigma_output = (2/m) * (self.sigma**2 - 1)
        # calculate KL error at W_sigma
        self.delta_KL_W_sigma = np.dot(self.enc_A_1.T, self.delta_KL_sigma_output) 
        
        # calculate the total error at W_mu
        self.dL_total_dW_mu = self.delta_recon_W_mu + (alpha_hyper_param * self.delta_KL_W_mu)
        # calculate total error at W_sigma
        self.dL_total_dW_sigma = self.delta_recon_W_sigma + (alpha_hyper_param * self.delta_KL_W_sigma)
        
        # calculate total error from the mu branch
        mu_branch = np.dot((self.delta_recon_latent_W_mu + (2*self.mu*alpha_hyper_param/m)), self.W_mu.T)
        # calculate total error from the sigma branch
        sigma_branch = np.dot(((self.delta_recon_latent_W_sigma * self.sigma) + ((2*alpha_hyper_param*(self.sigma**2 - 1)) / m)), self.W_sigma.T)
        # calculate total error at enc_W_1
        self.delta_enc_W_1 = np.dot(self.X.T, ((mu_branch + sigma_branch) * self._relu_derivative(self.enc_Z_1)))

        # calculate all of the bias errors
        self.delta_dec_B_L = np.sum(self.delta_recon_output, axis=0, keepdims=True)
        self.delta_dec_B_1 = np.sum(self.delta_decoder, axis=0, keepdims=True)
        self.delta_B_mu = np.sum(self.delta_recon_latent_W_mu + self.delta_KL_mu_output, axis=0, keepdims=True)
        self.delta_B_sigma = np.sum((self.delta_recon_latent_W_sigma * self.sigma) + self.delta_KL_sigma_output, axis=0, keepdims=True)
        self.delta_enc_B_1 = np.sum((mu_branch + sigma_branch) * self._relu_derivative(self.enc_Z_1), axis=0, keepdims=True)
    

    def GD_update_step(self):
        alpha_lr = self.learning_rate

        # 1. Update Encoder Weights & Biases
        self.enc_W_1 -= alpha_lr * self.delta_enc_W_1
        self.enc_B_1 -= alpha_lr * self.delta_enc_B_1

        # 2. Update Latent Weights & Biases
        self.W_mu -= alpha_lr * self.dL_total_dW_mu
        self.B_mu -= alpha_lr * self.delta_B_mu

        self.W_sigma -= alpha_lr * self.dL_total_dW_sigma
        self.B_sigma -= alpha_lr * self.delta_B_sigma

        # 3. Update Decoder Weights & Biases
        self.dec_W_1 -= alpha_lr * self.delta_recon_dec_W_1
        self.dec_B_1 -= alpha_lr * self.delta_dec_B_1

        self.dec_W_L -= alpha_lr * self.delta_recon_dec_W_L
        self.dec_B_L -= alpha_lr * self.delta_dec_B_L

    
    def _adam_update(self, matrix, gradient, v, s):
        """
        Helper function for the general logic for an update using the ADAM optimiser.
        
        :param matrix: Either the Weight or Bias to update
        :param gradient: The gradient at that weight or bias
        :param v: Momentum
        :param s: RMSProp
        """
        # 1. Update Momentum (v)
        v_new = (self.beta1 * v) + ((1 - self.beta1) * gradient)
        
        # 2. Update RMSProp (s)
        s_new = (self.beta2 * s) + ((1 - self.beta2) * (gradient ** 2))
        
        # 3. Update Weight/Bias
        matrix_new = matrix - self.learning_rate * (v_new / (np.sqrt(s_new) + self.adam_epsilon))
        
        return matrix_new, v_new, s_new
    

    def adam_step(self):
        """
        Actually performs the update to the weight and bias matrices by calling the _adam_update function.
        """
        # Encoder Updates
        self.enc_W_1, self.v_enc_W_1, self.s_enc_W_1 = self._adam_update(
            self.enc_W_1, self.delta_enc_W_1, self.v_enc_W_1, self.s_enc_W_1
        )
        self.enc_B_1, self.v_enc_B_1, self.s_enc_B_1 = self._adam_update(
            self.enc_B_1, self.delta_enc_B_1, self.v_enc_B_1, self.s_enc_B_1
        )

        # Latent Space Updates 
        self.W_mu, self.v_W_mu, self.s_W_mu = self._adam_update(
            self.W_mu, self.dL_total_dW_mu, self.v_W_mu, self.s_W_mu
        )
        self.B_mu, self.v_B_mu, self.s_B_mu = self._adam_update(
            self.B_mu, self.delta_B_mu, self.v_B_mu, self.s_B_mu
        )

        self.W_sigma, self.v_W_sigma, self.s_W_sigma = self._adam_update(
            self.W_sigma, self.dL_total_dW_sigma, self.v_W_sigma, self.s_W_sigma
        )
        self.B_sigma, self.v_B_sigma, self.s_B_sigma = self._adam_update(
            self.B_sigma, self.delta_B_sigma, self.v_B_sigma, self.s_B_sigma
        )

        # Decoder Updates
        self.dec_W_1, self.v_dec_W_1, self.s_dec_W_1 = self._adam_update(
            self.dec_W_1, self.delta_recon_dec_W_1, self.v_dec_W_1, self.s_dec_W_1
        )
        self.dec_B_1, self.v_dec_B_1, self.s_dec_B_1 = self._adam_update(
            self.dec_B_1, self.delta_dec_B_1, self.v_dec_B_1, self.s_dec_B_1
        )

        self.dec_W_L, self.v_dec_W_L, self.s_dec_W_L = self._adam_update(
            self.dec_W_L, self.delta_recon_dec_W_L, self.v_dec_W_L, self.s_dec_W_L
        )
        self.dec_B_L, self.v_dec_B_L, self.s_dec_B_L = self._adam_update(
            self.dec_B_L, self.delta_dec_B_L, self.v_dec_B_L, self.s_dec_B_L
        )
    

    def model_infer(self, num_samples):
        """
        Generates random noise vectors and visualises their decoded output.
        """
        # Generate random noise (z)
        z_samples = np.random.randn(num_samples, self.k_latent)
        
        # Forward pass through decoder layers
        dec_Z_1 = np.dot(z_samples, self.dec_W_1) + self.dec_B_1
        dec_A_1 = self._relu(dec_Z_1)
        
        dec_Z_L = np.dot(dec_A_1, self.dec_W_L) + self.dec_B_L
        x_decoded = self._sigmoid(dec_Z_L)
        
        # Plot images
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1: axes = [axes]
            
        for i, ax in enumerate(axes):
            digit = x_decoded[i].reshape(28, 28)
            ax.imshow(digit, cmap='Greys_r')
            ax.axis('off')
            ax.set_title(f"Sample {i+1}")
        
        plt.show()
    

    def save_model(self, filename):
        """
        Saves the model weights and configuration to a file.
        """
        directory = path.dirname(filename)
        if directory:
            makedirs(directory, exist_ok=True)

        model_data = {
            # Weights & Biases
            "enc_W_1": self.enc_W_1, 
            "enc_B_1": self.enc_B_1,
            "W_mu": self.W_mu,
            "B_mu": self.B_mu,
            "W_sigma": self.W_sigma,
            "B_sigma": self.B_sigma,
            "dec_W_1": self.dec_W_1,
            "dec_B_1": self.dec_B_1,
            "dec_W_L": self.dec_W_L,
            "dec_B_L": self.dec_B_L,
            
            # Hyperparameters
            "config": {
                "n": self.n,
                "hidden_dim": self.hidden_layer_dim,
                "k_latent": self.k_latent
            }
        }

        dump(model_data, filename)
        print(f"Model successfully saved to {filename}")