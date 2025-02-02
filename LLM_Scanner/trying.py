import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pickle

# For our RL environment and agent.
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import RobustScaler

# --- Indicator Calculation Function ---
def calculate_indicators(df):
    """Calculate technical indicators including VWAP and Fibonacci retracements."""
    try:
        stock_data = df.copy()

        # MACD
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        stock_data['SMA'] = stock_data['Close'].rolling(window=20, min_periods=2).mean()
        stock_data['std_dev'] = stock_data['Close'].rolling(window=20, min_periods=2).std()
        stock_data['Upper_Band'] = stock_data['SMA'] + (stock_data['std_dev'] * 2)
        stock_data['Lower_Band'] = stock_data['SMA'] - (stock_data['std_dev'] * 2)

        # RSI (using a 14-period window)
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        # VWAP
        stock_data['Typical_Price'] = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
        stock_data['Cum_Vol_Price'] = (stock_data['Typical_Price'] * stock_data['Volume']).cumsum()
        stock_data['Cum_Volume'] = stock_data['Volume'].cumsum()
        stock_data['VWAP'] = stock_data['Cum_Vol_Price'] / stock_data['Cum_Volume']

        # Fibonacci retracements (using overall min/max)
        high_price = stock_data['High'].max()
        low_price = stock_data['Low'].min()
        diff = high_price - low_price
        fib_levels = {
            'Fib_0.236': low_price + diff * 0.236,
            'Fib_0.382': low_price + diff * 0.382,
            'Fib_0.5':   low_price + diff * 0.5,
            'Fib_0.618': low_price + diff * 0.618,
            'Fib_0.786': low_price + diff * 0.786
        }
        for level, value in fib_levels.items():
            stock_data[level] = value

        # Fill missing values
        stock_data[['MACD','RSI','std_dev','SMA']] = stock_data[['MACD','RSI','std_dev','SMA']].fillna(0)
        stock_data['Close'] = stock_data['Close'].fillna(method='ffill')
        stock_data['Upper_Band'] = stock_data['Upper_Band'].fillna(stock_data['Close'] * 1.1)
        stock_data['Lower_Band'] = stock_data['Lower_Band'].fillna(stock_data['Close'] * 0.9)
        stock_data['VWAP'] = stock_data['VWAP'].fillna(stock_data['Close'])
        return stock_data
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return None

def get_stock_data(ticker, days, interval):
    """Get stock data with proper error handling and validation."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, multi_level_index=False)
        
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        stock_data.index = pd.to_datetime(stock_data.index)
        freq = 'H' if interval == '1h' else 'D' if interval == '1d' else 'W'
        stock_data = stock_data.asfreq(freq).fillna(method='ffill')
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

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
        df['rsi_buy'] = stock_data['RSI'] < 30  
        df['rsi_sell'] = stock_data['RSI'] > 70  
        df['bb_buy'] = stock_data['Close'] > stock_data['Lower_Band']
        df['bb_sell'] = stock_data['Close'] < stock_data['Upper_Band']
        df['fib_buy'] = stock_data['Close'] > stock_data['Fib_0.618']
        df['fib_sell'] = stock_data['Close'] < stock_data['Fib_0.382']
        
        # Combine signals
        buy_signals = (df['macd_buy'] | df['rsi_buy']) & (df['vwap_buy'] | df['bb_buy'] | df['fib_buy'])
        sell_signals = (df['macd_sell'] | df['rsi_sell']) & (df['vwap_sell'] | df['bb_sell'] | df['fib_sell'])
        
        signals = pd.Series(0, index=stock_data.index)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals.fillna(0)
    except Exception as e:
        print(f"Error generating signals: {str(e)}")
        return pd.Series(0, index=stock_data.index)
    
# --- 1. Create an RL Environment for Trading ---
class TradingEnv(gym.Env):
    """
    A simple trading environment.
    
    State: feature vector (technical indicators for a given timestep)
    Action: 0 = Hold, 1 = Buy, 2 = Sell.
    Reward: Change in net worth (capital + unrealized P/L).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_capital=100000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)  # work with integer-indexed rows
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # 0: no position, 1: long
        self.current_step = 0
        
        # Features used in the observation
        self.features = ['MACD', 'RSI', 'Close', 'Upper_Band', 'Lower_Band', 'VWAP', 'Signal_Line',
                         'Fib_0.618', 'Fib_0.382']
        self.scaler = RobustScaler()
        self.scaler.fit(self.data[self.features].values)
        obs_sample = self.scaler.transform(self.data[self.features].iloc[[0]].values)[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Initialize last_trade_price as first closing price
        self.last_trade_price = self.data.loc[0, 'Close']

    def reset(self):
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.position = 0
        self.last_trade_price = self.data.loc[0, 'Close']
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.data.loc[self.current_step, self.features].values.reshape(1, -1)
        obs_scaled = self.scaler.transform(obs)[0]
        return obs_scaled.astype(np.float32)
    
    def step(self, action):
        done = False
        current_price = self.data.loc[self.current_step, 'Close']
        # Calculate net worth before taking the action.
        net_worth_before = self.current_capital + (current_price - self.last_trade_price if self.position == 1 else 0)
        
        # Execute action:
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.last_trade_price = current_price
        elif action == 2:  # Sell
            if self.position == 1:
                profit = current_price - self.last_trade_price
                self.current_capital += profit
                self.position = 0
        # Else Hold (action == 0): do nothing.
        
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            new_price = current_price
        else:
            new_price = self.data.loc[self.current_step, 'Close']
        
        net_worth_after = self.current_capital + (new_price - self.last_trade_price if self.position == 1 else 0)
        reward = net_worth_after - net_worth_before
        
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    
    def render(self, mode='human', close=False):
        profit = self.current_capital - self.initial_capital
        print(f"Step: {self.current_step}, Capital: {self.current_capital:.2f}, Profit: {profit:.2f}")

# --- 2. Define the RL Training/Signal-Generation Flow ---
MODEL_FILE = "trading_rl_model.zip"

def generate_trading_signals_with_rl(stock_data, rl_iterations=10000):
    """
    Use an RL agent to generate trading signals.
    
    Flow:
      - Build an RL environment using stock data and technical indicators.
      - Load a pre-trained RL model if it exists; otherwise, initialize a new one.
      - Train for a fixed number of timesteps.
      - Roll out the policy over the data to generate signals.
      - Save the updated model.
    
    Returns:
      A Pandas Series of signals (1 = Buy, -1 = Sell, 0 = Hold).
    """
    # Ensure technical indicators exist.
    if 'MACD' not in stock_data.columns or 'RSI' not in stock_data.columns:
        stock_data = calculate_indicators(stock_data)
        if stock_data is None:
            raise ValueError("Error calculating indicators")
    
    for col in ['Fib_0.618', 'Fib_0.382']:
        if col not in stock_data.columns:
            stock_data = calculate_indicators(stock_data)
    
    # Create the RL environment.
    env = TradingEnv(stock_data)
    env = DummyVecEnv([lambda: env])
    
    # Load existing model or initialize new one.
    if os.path.exists(MODEL_FILE):
        model = DQN.load(MODEL_FILE, env=env)
        print("Loaded existing RL model.")
    else:
        model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3)
        print("Initialized new RL model.")
    
    # Train the agent.
    model.learn(total_timesteps=rl_iterations)
    model.save(MODEL_FILE)
    
    # Roll out the policy to generate signals.
    signals = []
    obs = env.reset()
    for _ in range(len(stock_data)):
        action, _ = model.predict(obs, deterministic=True)
        if action[0] == 1:
            signal = 1
        elif action[0] == 2:
            signal = -1
        else:
            signal = 0
        signals.append(signal)
        obs, _, done, _ = env.step(action)
        if done:
            break
    # Ensure the signals series covers the full date range.
    if len(signals) < len(stock_data):
        padding = [0] * (len(stock_data) - len(signals))
        signals.extend(padding)
    signals_series = pd.Series(signals, index=stock_data.index)
    return signals_series

def backtest_portfolio(stock_data, signals, initial_capital=100000):
    """
    A simple backtest to calculate portfolio value over time.
    
    Rules:
      - When a Buy signal occurs, go long (if not already in a position).
      - When a Sell signal occurs, exit long (if in a position).
      - Hold means no change.
      
    Returns:
      A DataFrame with portfolio value over time.
    """
    capital = initial_capital
    position = 0  # 0: no position, 1: long
    portfolio_values = []
    entry_price = 0
    
    n = min(len(stock_data), len(signals))
    for i in range(n):
        price = stock_data.iloc[i]['Close']
        signal = signals.iloc[i]
        if signal == 1 and position == 0:
            entry_price = price
            position = 1
        elif signal == -1 and position == 1:
            profit = price - entry_price
            capital += profit
            position = 0
        current_value = capital + (price - entry_price if position == 1 else 0)
        portfolio_values.append(current_value)
    
    return pd.DataFrame({'Portfolio_Value': portfolio_values}, index=stock_data.index[:n])

# --- Example Usage ---
if __name__ == "__main__":
    ticker = "INFY.NS"
    days = 365
    interval = '1d'
    
    stock_data = get_stock_data(ticker, days, interval)
    if stock_data is None:
        raise ValueError("No data retrieved for ticker.")
    stock_data = calculate_indicators(stock_data)
    
    # Flow 1: Traditional Signal Generation
    signals_traditional = generate_trading_signals(stock_data)
    print("Traditional Signals:")
    print(signals_traditional.tail())
    
    # Flow 2: RL-based Signal Generation
    signals_rl = generate_trading_signals_with_rl(stock_data, rl_iterations=100000)
    print("RL-based Signals:")
    print(signals_rl.tail())
    
    backtest_traditional = backtest_portfolio(stock_data, signals_traditional)
    backtest_rl = backtest_portfolio(stock_data, signals_rl)
    
    print("Final Portfolio Value (Traditional):", backtest_traditional.iloc[-1]['Portfolio_Value'])
    print("Final Portfolio Value (RL-based):", backtest_rl.iloc[-1]['Portfolio_Value'])
