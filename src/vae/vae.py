"""
Encoder_in (784) -> Encoder_hl_1 (512) -> Latent space (2/3) -> Decoder_hl_1 (512) -> Decoder_out (784)
"""

import numpy as np

class VAE:
    def __init__(self, n, hidden_layer_dim, k_latent, learning_rate=0.001):
        self.n = 784
        self.hidden_layer_dim = hidden_layer_dim
        self.k_latent = k_latent

        scale_input_weights = np.sqrt(2.0 / n)
        scale_hl_weights = np.sqrt(2.0 / hidden_layer_dim)
        scale_latent_input = np.sqrt(2.0 / k_latent)

        self.enc_W_1 = np.random.randn(n, hidden_layer_dim) * scale_input_weights
        self.enc_B_1 = np.zeros((1, hidden_layer_dim))

        self.W_mu = np.random.randn(hidden_layer_dim, k_latent) * scale_hl_weights
        self.B_mu = np.zeros((1, k_latent))

        self.W_sigma = np.random.randn(hidden_layer_dim, k_latent) * scale_hl_weights
        self.B_sigma = np.zeros((1, k_latent))

        self.dec_W_1 = np.random.randn(k_latent, hidden_layer_dim) * scale_latent_input
        self.dec_B_1 = np.zeros((1, hidden_layer_dim))

        self.dec_W_L = np.random.randn(hidden_layer_dim, n) * scale_hl_weights
        self.dec_B_L = np.zeros((1, n))

        self.learning_rate = learning_rate


    def relu(self, matrix):
        return np.maximum(0, matrix)

    def sigmoid(self, matrix):
        return 1 / (1 + np.exp(-matrix))


    def forward_pass(self, X):
        """
        Performs one forward pass.
        
        :param X: Input matrix of shape (Batch_Size, 784) = (m, n)
        """
        self.X = X
        # Encoder
        self.enc_Z_1 = np.dot(self.X, self.enc_W_1) + self.enc_B_1 # (m, n) x (n, k_1) + (m, k_1) = (m, k_1)
        self.enc_A_1 = self.relu(self.enc_Z_1) # (m, k_1)

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
        self.dec_A_1 = self.relu(self.dec_Z_1) # (m, k_1)

        self.dec_Z_L = np.dot(self.dec_A_1, self.dec_W_L) + self.dec_B_L # (m, k_1) x (k_1, n) + (m, n) = (m, n)
        self.dec_A_L = self.sigmoid(self.dec_Z_L) # (m, n)
        
        self.X_hat = self.dec_A_L # (m, n)

        return self.X_hat, self.mu, self.sigma