import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker="SYNTH-TECH", days=1000):
        self.ticker = ticker
        self.days = days
        self.df = None
        self.model = None

    def generate_synthetic_data(self):
        """
        Generates synthetic stock price data using Geometric Brownian Motion.
        This mimics real-world stock volatility and trends.
        """
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=self.days, freq='B') # Business days
        
        # Parameters for Geometric Brownian Motion
        mu = 0.0005  # Drift (expected return)
        sigma = 0.02 # Volatility
        dt = 1       # Time step
        
        # Generate prices
        prices = [150.0] # Starting price
        for _ in range(len(dates) - 1):
            price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
            prices.append(price)
            
        # Create DataFrame
        self.df = pd.DataFrame(data={'Date': dates, 'Close': prices})
        self.df['Volume'] = np.random.randint(100000, 5000000, size=self.days)
        
        print(f"--- Generated {self.days} days of data for {self.ticker} ---")
        return self.df

    def add_technical_indicators(self):
        """
        Feature Engineering: Adds Moving Averages and Daily Returns.
        """
        df = self.df
        
        # 7-day and 21-day Simple Moving Averages
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        
        # Daily Returns (Percentage change)
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Volatility (Rolling Standard Deviation)
        df['Volatility'] = df['Close'].rolling(window=7).std()
        
        # Target Variable: 'Next_Close' (The price we want to predict)
        # We shift the Close price back by 1 day to align "today's data" with "tomorrow's price"
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values created by rolling windows and shifting
        self.df = df.dropna()
        print("--- Technical Indicators Added ---")

    def train_model(self):
        """
        Trains a Random Forest Regressor.
        """
        print("\n--- Training Model ---")
        
        # Features (X) and Target (y)
        features = ['Close', 'Volume', 'SMA_7', 'SMA_21', 'Daily_Return', 'Volatility']
        X = self.df[features]
        y = self.df['Target']
        
        # Split data (Shuffle=False is important for time series!)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluation
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f"Model Performance:")
        print(f"MAE (Mean Absolute Error): ${mae:.2f}")
        print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
        
        return X_test, y_test, predictions

    def visualize_results(self, y_test, predictions):
        """
        Plots the Actual vs Predicted stock prices.
        """
        plt.figure(figsize=(14, 7))
        
        # Create a timeline for the test set
        test_dates = self.df.iloc[-len(y_test):]['Date']
        
        plt.plot(test_dates, y_test, label='Actual Price', color='blue', linewidth=2)
        plt.plot(test_dates, predictions, label='Predicted Price', color='orange', linestyle='--', linewidth=2)
        
        plt.title(f'{self.ticker} Stock Price Prediction (Random Forest)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # In a script, we use show(), but for saving you can use savefig()
        print("\n[Info] Displaying visualization...")
        plt.show()

def main():
    # Initialize Project
    predictor = StockPredictor(ticker="AI-CORP")
    
    # 1. Get Data
    predictor.generate_synthetic_data()
    
    # 2. Feature Engineering
    predictor.add_technical_indicators()
    
    # 3. Train & Evaluate
    X_test, y_test, preds = predictor.train_model()
    
    # 4. Visualize
    predictor.visualize_results(y_test, preds)
    
    # Optional: Show Feature Importance
    importances = predictor.model.feature_importances_
    features = ['Close', 'Volume', 'SMA_7', 'SMA_21', 'Daily_Return', 'Volatility']
    print("\nTop Predictors:")
    for feat, imp in zip(features, importances):
        print(f"{feat}: {imp:.4f}")

if __name__ == "__main__":
    main()