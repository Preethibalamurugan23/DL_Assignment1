# visualising dataset
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb
import numpy as np
import pandas as pd
import wandb
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Train Neural Network with Fashion-MNIST")
parser.add_argument("--wandb_entity", type=str, required=True, help="WandB entity (username or team name)")
parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name")

# WandB Arguments
parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name for Weights & Biases tracking.")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="WandB entity for tracking experiments.")

# Dataset Selection
parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use.")

# Training Hyperparameters
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for training.")

# Loss Function
parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function.")

# Optimizer Selection
parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer choice.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum (for momentum and nag optimizers).")
parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta (for RMSprop).")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 (for Adam & Nadam).")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 (for Adam & Nadam).")
parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for numerical stability in optimizers.")
parser.add_argument("-w_d", "--weight_decay", type=float, default=0.5, help="Weight decay for optimizers.")

# Model Architecture
parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method.")
parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers.")
parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons in each hidden layer.")
parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function.")
args = parser.parse_args()

wandb.init(entity=args.wandb_entity, project=args.wandb_project, name="Model training")


class Config:
    def __init__(self, args):
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.loss = args.loss
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.beta = args.beta
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.weight_decay = args.weight_decay
        self.weight_init = args.weight_init
        self.hidden_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.activation = args.activation

config = Config(args)

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)  # Flatten

# One-hot encode labels
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# Split into training (90%) and validation (10%)
val_size = int(0.1 * x_train.shape[0])
x_val, y_val_onehot = x_train[:val_size], y_train_onehot[:val_size]
x_train, y_train_onehot = x_train[val_size:], y_train_onehot[val_size:]

def initialize_weights(shape, method):
    """Initialize weights using Xavier or small random values."""
    if method == "xavier":
        return np.random.randn(*shape) * np.sqrt(1 / shape[0])
    else:
        return np.random.randn(*shape) * 0.01

class NeuralNetwork:
    def __init__(self, config):
        input_size = 784  # MNIST images are 28x28
        output_size = 10   # 10 classes (digits 0-9)
        self.layers = [input_size] + [config.hidden_size] * config.hidden_layers + [output_size]
        self.activation_func = config.activation
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.beta1 = 0.5
        self.beta2 = 0.5
        self.momentum = 0.5
        self.epsilon = 0.000001	
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        for i in range(len(self.layers) - 1):
            self.weights[i] = initialize_weights((self.layers[i], self.layers[i+1]), config.weight_init)
            self.biases[i] = np.zeros((1, self.layers[i+1]))
            
        self.velocities = {i: np.zeros_like(self.weights[i]) for i in self.weights}
        self.squared_gradients = {i: np.zeros_like(self.weights[i]) for i in self.weights}
        self.m_t = {i: np.zeros_like(self.weights[i]) for i in self.weights}
        self.v_t = {i: np.zeros_like(self.weights[i]) for i in self.weights}

        

    def activation(self, Z):
        """Applies activation function."""
        if self.activation_func == "relu":
            return np.maximum(0, Z)
        elif self.activation_func == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation_func == "tanh":
            return np.tanh(Z)

    def activation_derivative(self, Z):
        """Computes derivative of activation function based on pre-activation Z."""
        if self.activation_func == "relu":
            return (Z > 0).astype(float)  # Derivative of ReLU is 1 for Z > 0, else 0
        elif self.activation_func == "sigmoid":
            A = 1 / (1 + np.exp(-Z))  # Compute sigmoid(Z)
            return A * (1 - A)  # Correct derivative
        elif self.activation_func == "tanh":
            A = np.tanh(Z)  # Compute tanh(Z)
            return 1 - A**2  # Correct derivative


    def softmax(self, Z):
        """Numerically stable softmax function."""
        Z -= np.max(Z, axis=1, keepdims=True)  # Prevent overflow
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        """Performs forward propagation."""
        activations = {0: X}
        pre_activations = {}  # Store pre-activation values (Z)

        for i in range(len(self.layers) - 2):
            Z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            pre_activations[i + 1] = Z  # Save Z before activation
            activations[i + 1] = self.activation(Z)

        # Output layer with softmax
        Z_out = np.dot(activations[len(self.layers) - 2], self.weights[len(self.layers) - 2]) + self.biases[len(self.layers) - 2]
        pre_activations[len(self.layers) - 1] = Z_out  # Save final layer pre-activation
        activations[len(self.layers) - 1] = self.softmax(Z_out)

        return activations, pre_activations

    def backward(self, activations, pre_activations, Y_true):
        grads = {}
        L = len(self.layers) - 1
        m = Y_true.shape[0]

        # Compute gradient for output layer
        dZ = activations[L] - Y_true
        grads['dW' + str(L - 1)] = np.dot(activations[L - 1].T, dZ) / m
        grads['db' + str(L - 1)] = np.sum(dZ, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(L - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)
            dZ = dA * self.activation_derivative(pre_activations[i + 1])  # Correct usage of Z
            grads['dW' + str(i)] = np.dot(activations[i].T, dZ) / m
            grads['db' + str(i)] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads

    def update_weights(self, grads, t):
        for i in self.weights:
            before_update = np.linalg.norm(self.weights[i])  # Compute norm before update

            if self.optimizer == "sgd":
                self.weights[i] -= self.learning_rate * grads['dW' + str(i)]
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

            elif self.optimizer == "momentum":
                self.velocities[i] = self.beta1 * self.velocities[i] - self.learning_rate * grads['dW' + str(i)]
                self.weights[i] += self.velocities[i]
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

            elif self.optimizer == "nesterov":
                prev_velocity = self.velocities[i]
                self.velocities[i] = self.beta1 * self.velocities[i] - self.learning_rate * grads['dW' + str(i)]
                self.weights[i] += -self.beta1 * prev_velocity + (1 + self.beta1) * self.velocities[i]
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

            elif self.optimizer == "rmsprop":
                self.squared_gradients[i] = self.beta2 * self.squared_gradients[i] + (1 - self.beta2) * (grads['dW' + str(i)] ** 2)
                self.weights[i] -= self.learning_rate * grads['dW' + str(i)] / (np.sqrt(self.squared_gradients[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

            elif self.optimizer == "adam":
                self.m_t[i] = self.beta1 * self.m_t[i] + (1 - self.beta1) * grads['dW' + str(i)]
                self.v_t[i] = self.beta2 * self.v_t[i] + (1 - self.beta2) * (grads['dW' + str(i)] ** 2)
                m_hat = self.m_t[i] / (1 - self.beta1 ** t)
                v_hat = self.v_t[i] / (1 - self.beta2 ** t)
                self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

            elif self.optimizer == "nadam":
                m_hat = self.beta1 * self.m_t[i] + (1 - self.beta1) * grads['dW' + str(i)]
                v_hat = self.beta2 * self.v_t[i] + (1 - self.beta2) * (grads['dW' + str(i)] ** 2)
                self.weights[i] -= self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * grads['dW' + str(i)] / (1 - self.beta1 ** t)) / (np.sqrt(v_hat / (1 - self.beta2 ** t)) + self.epsilon)
                self.biases[i] -= self.learning_rate * grads['db' + str(i)]

    def cross_entropy(self, Y_true, Y_pred):
        return -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-9), axis=1))

    def cross_entropy_derivative(self, Y_true, Y_pred):
        return Y_pred - Y_true

    def mse(self, Y_true, Y_pred):
        return np.mean((Y_true - Y_pred) ** 2)

    def mse_derivative(self, Y_true, Y_pred):
        return (Y_pred - Y_true) / Y_true.shape[0]

    def compute_loss(self, Y_true, Y_pred):
        """Computes categorical cross-entropy loss."""
        loss = self.cross_entropy(Y_true, Y_pred) if self.loss_function == "cross_entropy" else self.mse(Y_true, Y_pred)
        return loss

    def compute_accuracy(self, Y_true, Y_pred):
        """Computes accuracy given true labels and predicted probabilities."""
        Y_pred_labels = np.argmax(Y_pred, axis=1)
        Y_true_labels = np.argmax(Y_true, axis=1)
        return np.mean(Y_pred_labels == Y_true_labels)

    def train(self):
        """Train the neural network and log metrics using wandb."""
        for epoch in range(self.epochs):
            # Forward pass
            activations, pre_activations = self.forward(x_train)
            train_loss = self.compute_loss(y_train_onehot, activations[len(self.layers) - 1])
            train_accuracy = self.compute_accuracy(y_train_onehot, activations[len(self.layers) - 1])

            # Backward pass & update
            grads = self.backward(activations, pre_activations, y_train_onehot)
            self.update_weights(grads, epoch + 1)

            # Validation metrics
            val_activations, _ = self.forward(x_val)
            val_loss = self.compute_loss(y_val_onehot, val_activations[len(self.layers) - 1])
            val_accuracy = self.compute_accuracy(y_val_onehot, val_activations[len(self.layers) - 1])

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "training_loss": train_loss,
                "training_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

model = NeuralNetwork(config)
model.train()