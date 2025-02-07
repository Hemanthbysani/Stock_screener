import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from calculate_indicators import calculate_indicators
import numpy as np
import backtest
import gym
from gym import spaces
import os
import joblib
from datetime import datetime

def generate_trading_signals(stock_data):
    """Generate trading signals using MACD, RSI, Bollinger Bands, VWAP, and Fibonacci retracements."""
    try:
        df = pd.DataFrame(index=stock_data.index)
        
        # Primary signals (the "core" indicators)
        df['macd_buy'] = stock_data['MACD'] > stock_data['Signal_Line']
        df['macd_sell'] = stock_data['MACD'] < stock_data['Signal_Line']
        df['vwap_buy'] = stock_data['Close'] > stock_data['VWAP']
        df['vwap_sell'] = stock_data['Close'] < stock_data['VWAP']
        
        # Secondary signals (optional confirmations)
        # Using optimized RSI thresholds: <30 (oversold) for buys, >70 (overbought) for sells.
        df['rsi_buy'] = stock_data['RSI'] < 30  
        df['rsi_sell'] = stock_data['RSI'] > 70  
        
        # Bollinger Bands: Price above lower band (support) for buy; below upper band for sell.
        df['bb_buy'] = stock_data['Close'] > stock_data['Lower_Band']
        df['bb_sell'] = stock_data['Close'] < stock_data['Upper_Band']
        
        # Fibonacci: For a stronger confirmation, we require the price to be above the 61.8% level for buys 
        # and below the 38.2% level for sells.
        df['fib_buy'] = stock_data['Close'] > stock_data['Fib_0.618']
        df['fib_sell'] = stock_data['Close'] < stock_data['Fib_0.382']
        
        # Combine the signals.
        # Here we require that the "core" indicators (MACD and VWAP) agree,
        # and that at least one of the secondary signals (RSI, Bollinger Bands, or Fibonacci) confirms.
        buy_signals = (df['macd_buy'] | df['rsi_buy']) & (df['vwap_buy'] | df['bb_buy'] | df['fib_buy'])
        sell_signals = (df['macd_sell'] | df['rsi_sell']) & (df['vwap_sell'] | df['bb_sell'] | df['fib_sell'])
        
        signals = pd.Series(0, index=stock_data.index)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals.fillna(0)
    except Exception as e:
        print(f"Error generating signals: {str(e)}")
        return pd.Series(0, index=stock_data.index)

# def generate_trading_signals_with_ml(stock_data):
#     """Generate trading signals using machine learning and evaluate via backtest"""
#     try:
#         # Ensure the key indicators are available.
#         if 'MACD' not in stock_data.columns or 'RSI' not in stock_data.columns:
#             stock_data = calculate_indicators(stock_data)
#             if stock_data is None:
#                 raise ValueError("Error calculating indicators")
        
#         # Build the feature DataFrame (as before)
#         df = pd.DataFrame(index=stock_data.index)
#         df['MACD'] = stock_data['MACD']
#         df['RSI'] = stock_data['RSI']
#         df['Close'] = stock_data['Close']
#         df['Upper_Band'] = stock_data['Upper_Band']
#         df['Lower_Band'] = stock_data['Lower_Band']
#         df['VWAP'] = stock_data['VWAP']
#         df['Signal_Line'] = stock_data['Signal_Line']
#         df['Fib_0.618'] = stock_data['Fib_0.618']
#         df['Fib_0.382'] = stock_data['Fib_0.382']
        
#         # Also include the primary and secondary technical signals (for features)
#         df['macd_buy'] = stock_data['MACD'] > stock_data['Signal_Line']
#         df['macd_sell'] = stock_data['MACD'] < stock_data['Signal_Line']
#         df['vwap_buy'] = stock_data['Close'] > stock_data['VWAP']
#         df['vwap_sell'] = stock_data['Close'] < stock_data['VWAP']
#         df['rsi_buy'] = stock_data['RSI'] < 30  
#         df['rsi_sell'] = stock_data['RSI'] > 70  
#         df['bb_buy'] = stock_data['Close'] > stock_data['Lower_Band']
#         df['bb_sell'] = stock_data['Close'] < stock_data['Upper_Band']
#         df['fib_buy'] = stock_data['Close'] > stock_data['Fib_0.618']
#         df['fib_sell'] = stock_data['Close'] < stock_data['Fib_0.382']
        
#         # Instead of computing targets via simulation for every step,
#         # we now intend to generate trading signals (targets) directly using the ML model.
#         #
#         # Prepare feature matrix X.
#         feature_cols = [
#             'MACD', 'RSI', 'Close', 'Upper_Band', 'Lower_Band',
#             'VWAP', 'Signal_Line', 'Fib_0.618', 'Fib_0.382',
#             'macd_buy', 'macd_sell', 'vwap_buy', 'vwap_sell', 
#             'rsi_buy', 'rsi_sell', 'bb_buy', 'bb_sell', 'fib_buy', 'fib_sell'
#         ]
#         X = df[feature_cols]
        
#         # Create targets using traditional technical analysis signals
#         target = np.where((df['macd_buy'] & df['vwap_buy']), 1, 
#              np.where((df['macd_sell'] & df['vwap_sell']), -1, 0))
#         df['target'] = pd.Series(target, index=stock_data.index)
#         y = df['target']
#         # Chronologically split data into training and test sets first
#         split_idx = int(len(df) * 0.7)
#         X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#         y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
#         # Scale features using only training data
#         scaler = RobustScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#         # Train the classifier.
#         model = RandomForestClassifier(random_state=42, verbose=0, n_estimators=150, criterion='log_loss')
#         model.fit(X_train_scaled, y_train)
        
#         # Evaluate on the test set.
#         y_pred = model.predict(X_test_scaled)
#         y_pred = model.predict(X_test)
#         print(y_pred)
#         print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))
#         # Generate signals using a rolling window approach
#         signals = []
#         for i in range(len(X)):
#             if i < split_idx:
#                 signals.append(0)  # No trading during training period
#             else:
#                 current_features = scaler.transform(X.iloc[i:i+1])
#                 signals.append(model.predict(current_features)[0])
#         signals_series = pd.Series(signals, index=df.index).fillna(0)
        
#         # Commented out the backtest call to avoid infinite loop.
#         # initial_balance = 10000
#         # sim_result = backtest.backtest(stock_data, initial_balance=initial_balance, strategy='ml')
#         # print("Backtest result with ML-generated signals:", sim_result)
        
#         return signals_series
#     except Exception as e:
#         print(f"Error generating signals with ML_Model: {str(e)}")
#         return pd.Series(0, index=stock_data.index)
    
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, window_size=10):
        super().__init__()
        self.data = data.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * len(self.data.columns),), 
            dtype=np.float32
        )
        self.current_step = window_size
        self.signals = np.zeros(len(self.data), dtype=int)

    def _get_obs(self):
        # Only use historical data up to current step
        obs_window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        return obs_window.values.flatten().astype(np.float32)

    def step(self, action):
        if self.current_step < len(self.data):
            if action == 1:
                self.signals[self.current_step] = 1
            elif action == 2:
                self.signals[self.current_step] = -1

        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        reward = 0
        if done:
            # Only evaluate trades up to the current point
            metrics = backtest.backtest(
                pd.Series(self.signals[:self.current_step]),
                self.data.iloc[:self.current_step],
                initial_balance=self.initial_balance
            )
            reward = metrics['final_value'] - self.initial_balance
            
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.current_step = self.window_size
        self.signals = np.zeros(len(self.data), dtype=int)
        return self._get_obs()

def generate_trading_signals_with_ml(stock_data, ticker, initial_balance=10000):
    """Generate trading signals using reinforcement learning"""
    try:
        
        episodes = 100
        # Ensure all required indicators are available
        if 'MACD' not in stock_data.columns:
            stock_data = calculate_indicators(stock_data)
            if stock_data is None:
                raise ValueError("Error calculating indicators")
        
        # Create models directory if it doesn't exist
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Create the environment with the complete dataset
        env = TradingEnv(stock_data, initial_balance=initial_balance)
        best_signals = None
        best_reward = float('-inf')
        best_env = None
        
        # Train the agent for multiple episodes
        for _ in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # Replace with actual RL agent action
                obs, reward, done, info = env.step(action)
            
            # Keep track of best performing signals
            if done and reward > best_reward:
                print("Rewards: ", reward)
                best_reward = reward
                best_signals = env.signals.copy()
                best_env = env
        
        # Save the best model with ticker and datetime
        if best_env is not None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M")
            model_filename = f"{models_dir}/{ticker}_{current_time}_model.joblib"
            model_data = {
                'env': best_env,
                'signals': best_signals,
                'reward': best_reward,
                'ticker': ticker,
                'date': current_time
            }
            joblib.dump(model_data, model_filename)
            print(f"Model saved as: {model_filename}")
        
        # Return the best performing signals
        if best_signals is not None:
            return pd.Series(best_signals, index=stock_data.index)
        return pd.Series(0, index=stock_data.index)
        
    except Exception as e:
        print(f"Error generating signals with RL: {str(e)}")
        return pd.Series(0, index=stock_data.index)
    
def predict_with_saved_model(stock_data, ticker, model_dir='models'):
    """Use the most recent saved model to predict trading signals"""
    try:
        # Find the most recent model for this ticker
        model_files = [f for f in os.listdir(model_dir) if f.startswith(ticker) and f.endswith('_model.joblib')]
        if not model_files:
            raise ValueError(f"No saved models found for ticker {ticker}")
        
        latest_model_file = max(model_files, key=lambda x: x.split('_')[1])
        model_path = os.path.join(model_dir, latest_model_file)
        
        # Load the saved model
        model_data = joblib.load(model_path)
        env = model_data['env']
        
        # Generate predictions
        signals = np.zeros(len(stock_data))
        current_step = env.window_size
        
        while current_step < len(stock_data):
            obs_window = stock_data.iloc[current_step - env.window_size:current_step]
            obs = obs_window.values.flatten().astype(np.float32)
            action = env.action_space.sample()  # Replace with actual model prediction
            if action == 1:
                signals[current_step] = 1
            elif action == 2:
                signals[current_step] = -1
            current_step += 1
            
        return pd.Series(signals, index=stock_data.index)
        
    except Exception as e:
        print(f"Error predicting with saved model: {str(e)}")
        return pd.Series(0, index=stock_data.index)
