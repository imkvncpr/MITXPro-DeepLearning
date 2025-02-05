from io import open
import unicodedata
import string
import random
import re
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
import plotly.graph_objects as go
import logging
from typing import Tuple, Dict, Optional, Union
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    def __init__(self, 
                 ticker: str = 'BTC-USD',
                 look_back: int = 60,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the cryptocurrency price predictor.
        
        Args:
            ticker: Trading symbol for the cryptocurrency
            look_back: Number of previous time steps to use for prediction
            test_size: Proportion of dataset to use for testing
            random_state: Random seed for reproducibility
        """
        self.ticker = ticker
        self.look_back = look_back
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X = None
        
    def download_crypto_data(self, years: int = 5) -> Optional[pd.DataFrame]:
        """
        Download historical cryptocurrency data.
        
        Args:
            years: Number of years of historical data to download
            
        Returns:
            DataFrame containing historical price data
        """
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            data = yf.download(self.ticker, start=start_date, end=end_date)
            logger.info(f"Successfully downloaded {years} years of {self.ticker} data")
            return data
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model training.
        
        Args:
            data: Raw price data
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        try:
            # Use more features: Open, High, Low, Close, Volume
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data_processed = data[feature_columns].values
            
            # Scale features independently
            scaled_data = self.scaler.fit_transform(data_processed)
            
            X, y = [], []
            for i in range(self.look_back, len(scaled_data)):
                X.append(scaled_data[i-self.look_back:i])
                y.append(scaled_data[i, 3])  # Index 3 corresponds to Close price
            
            X, y = np.array(X), np.array(y)
            logger.info(f"Prepared data shapes: X: {X.shape}, y: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def create_model(self, 
                    neurons: int = 100,
                    dropout: float = 0.3) -> Sequential:
        """
        Create LSTM model architecture.
        
        Args:
            neurons: Number of LSTM units in first layer
            dropout: Dropout rate for regularization
            
        Returns:
            Compiled Keras Sequential model
        """
        try:
            model = Sequential([
                LSTM(neurons, return_sequences=True, 
                     input_shape=(self.look_back, 5)),  # 5 features
                Dropout(dropout),
                LSTM(neurons//2, return_sequences=False),
                Dropout(dropout/2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def train_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   optimize_hyperparameters: bool = False,
                   **kwargs) -> Tuple[Sequential, Dict[str, float]]:
        """
        Train the LSTM model with optional hyperparameter optimization.
        
        Args:
            X: Input features
            y: Target values
            optimize_hyperparameters: Whether to perform hyperparameter tuning
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(X_train, y_train)
            else:
                model = self.create_model()
                
                # Add callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
                ]
                
                model.fit(
                    X_train, y_train,
                    epochs=kwargs.get('epochs', 100),
                    batch_size=kwargs.get('batch_size', 32),
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Evaluate model
            predictions = model.predict(X_test, verbose=0)
            metrics = self._calculate_metrics(y_test, predictions)
            
            self.model = model
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
        """
        Perform hyperparameter optimization using RandomizedSearchCV.
        """
        model = KerasRegressor(model=self.create_model, verbose=0)
        
        param_dist = {
            'model__neurons': [50, 100, 150, 200],
            'model__dropout': [0.2, 0.3, 0.4, 0.5],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150]
        }
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=2
        )
        
        random_search.fit(X_train, y_train)
        logger.info(f"Best hyperparameters: {random_search.best_params_}")
        return random_search.best_estimator_.model_

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        """
        # Inverse transform predictions to original scale
        y_true_orig = self.scaler.inverse_transform(np.column_stack([np.zeros((len(y_true), 3)), 
                                                                    y_true.reshape(-1, 1),
                                                                    np.zeros((len(y_true), 1))]))[:, 3]
        y_pred_orig = self.scaler.inverse_transform(np.column_stack([np.zeros((len(y_pred), 3)), 
                                                                    y_pred.reshape(-1, 1),
                                                                    np.zeros((len(y_pred), 1))]))[:, 3]
        
        return {
            'MSE': mean_squared_error(y_true_orig, y_pred_orig),
            'RMSE': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'MAE': mean_absolute_error(y_true_orig, y_pred_orig),
            'R2': r2_score(y_true_orig, y_pred_orig)
        }

    def visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Create interactive visualization of predictions vs actual values.
        """
        # Inverse transform the data
        y_true_orig = self.scaler.inverse_transform(np.column_stack([np.zeros((len(y_true), 3)), 
                                                                    y_true.reshape(-1, 1),
                                                                    np.zeros((len(y_true), 1))]))[:, 3]
        y_pred_orig = self.scaler.inverse_transform(np.column_stack([np.zeros((len(y_pred), 3)), 
                                                                    y_pred.reshape(-1, 1),
                                                                    np.zeros((len(y_pred), 1))]))[:, 3]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true_orig, mode='lines', name='Actual Prices'))
        fig.add_trace(go.Scatter(y=y_pred_orig, mode='lines', name='Predicted Prices'))
        fig.update_layout(
            title=f'{self.ticker} Price Prediction',
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_dark'
        )
        fig.show()

    # [Previous imports and class methods remain the same until predict_future...]

    def predict_future(self, days: int = 30) -> np.ndarray:
        """
        Predict future prices for a specified number of days.
        
        Args:
            days: Number of days to predict into the future
            
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            # Get the most recent data window
            current_sequence = self.X[-1:].copy()  # Shape: (1, look_back, 5)
            future_predictions = []
            
            for _ in range(days):
                # Make prediction for next day
                next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]  # Get scalar value
                future_predictions.append(next_pred)
                
                # Shift the sequence and update with new prediction
                # Move all timesteps one step back
                current_sequence = current_sequence.copy()
                current_sequence[0, :-1] = current_sequence[0, 1:]
                
                # Create new timestep data
                new_timestep = current_sequence[0, -1].copy()
                new_timestep[3] = next_pred  # Update only the Close price
                
                # Update the last timestep
                current_sequence[0, -1] = new_timestep
                
            # Convert predictions back to original scale
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions_reshaped = np.hstack([
                np.zeros((len(future_predictions), 3)),  # Open, High, Low
                future_predictions,                      # Close
                np.zeros((len(future_predictions), 1))   # Volume
            ])
            
            future_prices = self.scaler.inverse_transform(future_predictions_reshaped)[:, 3]
            
            logger.info(f"Generated {days} days of future predictions")
            return future_prices
            
        except Exception as e:
            logger.error(f"Error making future predictions: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the cryptocurrency price prediction pipeline.
    """
    # Initialize predictor
    predictor = CryptoPricePredictor(ticker='BTC-USD', look_back=60)
    
    try:
        # Download and prepare data
        logger.info("Downloading cryptocurrency data...")
        data = predictor.download_crypto_data(years=5)
        if data is None:
            logger.error("Failed to download data")
            return
        
        # Prepare data
        logger.info("Preparing data for training...")
        X, y = predictor.prepare_data(data)
        predictor.X = X  # Store for future predictions
        
        # Train model
        logger.info("Training the model...")
        model, metrics = predictor.train_model(
            X, y,
            optimize_hyperparameters=False,
            epochs=100,
            batch_size=32
        )
        
        # Print metrics
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        
        # Visualize historical predictions
        logger.info("Generating historical predictions visualization...")
        y_pred = model.predict(X, verbose=0)
        predictor.visualize_predictions(y, y_pred)
        
        # Make future predictions
        logger.info("Generating future price predictions...")
        future_prices = predictor.predict_future(days=30)
        print("\nPredicted prices for next 30 days:")
        for i, price in enumerate(future_prices, 1):
            print(f"Day {i}: ${price:,.2f}")

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
