# -*- coding: utf-8 -*-
"""Deep Learning: Mastering Neural Networks - Module 2 Assignment: Wine Classification"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import os
from sklearn.preprocessing import StandardScaler

def download_wine_data():
    """Download wine quality datasets if they don't exist locally"""
    urls = {
        'red': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        'white': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    }
    
    for wine_type, url in urls.items():
        filename = f'winequality-{wine_type}.csv'
        if not os.path.exists(filename):
            print(f"Downloading {wine_type} wine dataset...")
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {wine_type} wine dataset: {str(e)}")
                raise
        else:
            print(f"{filename} already exists, skipping download")

def load_and_prepare_data():
    """Load and prepare the wine datasets"""
    try:
        # Ensure data files exist
        download_wine_data()
        
        # Read and prepare red wine data
        df_red = pd.read_csv('winequality-red.csv', delimiter=";")
        df_red["color"] = 1  # 1 for red wine
        
        # Read and prepare white wine data
        df_white = pd.read_csv('winequality-white.csv', delimiter=";")
        df_white["color"] = 0  # 0 for white wine
        
        # Combine and shuffle datasets
        df = pd.concat([df_red, df_white])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Select features and prepare numpy arrays
        input_columns = ["citric acid", "residual sugar", "total sulfur dioxide"]
        output_columns = ["color"]
        
        X = df[input_columns].to_numpy()
        Y = df[output_columns].to_numpy()
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, Y
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise

class SingleNeuronClassifier:
    def __init__(self, input_dim):
        # Initialize weights and bias with small random values
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = np.random.randn() * 0.01
        self.training_loss = []
    
    def forward(self, x):
        # Forward pass using sigmoid activation
        return self.sigmoid(np.dot(x, self.weights) + self.bias)
    
    def sigmoid(self, z):
        # Sigmoid activation function with clipping to prevent overflow
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy(self, y_true, y_pred):
        # Compute binary cross entropy loss
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def train(self, X, Y, learning_rate=0.001, epochs=200, batch_size=32):
        num_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data at start of each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_Y = Y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(batch_X)
                
                # Compute gradients
                error = y_pred - batch_Y.reshape(-1)
                dw = np.mean(error.reshape(-1, 1) * batch_X, axis=0)
                db = np.mean(error)
                
                # Update weights and bias
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
                
                # Accumulate loss
                batch_loss = self.binary_cross_entropy(batch_Y, y_pred)
                epoch_loss += batch_loss * len(batch_X)
            
            # Average loss for the epoch
            epoch_loss /= num_samples
            self.training_loss.append(epoch_loss)
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

def evaluate_classification_accuracy(model, input_data, labels):
    """Evaluate model accuracy and print results"""
    predictions = model.forward(input_data)
    predicted_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_labels == labels.reshape(-1))
    
    print(f"Model predicted {accuracy * 100:.2f}% of samples correctly")
    return accuracy

def plot_training_loss(model):
    """Plot the training loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(model.training_loss)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.grid(True)
    plt.show()

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X, Y = load_and_prepare_data()
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    print("\nInitializing and training model...")
    model = SingleNeuronClassifier(X.shape[1])
    model.train(X_train, Y_train, learning_rate=0.001, epochs=200, batch_size=32)
    
    # Evaluate model
    print("\nTraining Set Performance:")
    train_accuracy = evaluate_classification_accuracy(model, X_train, Y_train)
    print("\nTest Set Performance:")
    test_accuracy = evaluate_classification_accuracy(model, X_test, Y_test)
    
    # Plot training loss
    plot_training_loss(model)

if __name__ == "__main__":
    main()