import pandas as pd
import numpy as np

def calculate_returns(prices):
    """Calculate returns from price series"""
    return prices.pct_change()

def calculate_portfolio_value(signals, prices, initial_balance=10000):
    """Calculate portfolio value based on signals"""
    position = 0
    balance = initial_balance
    portfolio_values = []
    
    for i in range(len(signals)):
        if signals[i] == 1 and position == 0:  # Buy
            position = balance / prices[i]
            balance = 0
        elif signals[i] == -1 and position > 0:  # Sell
            balance = position * prices[i]
            position = 0
            
        current_value = balance + (position * prices[i] if position > 0 else 0)
        portfolio_values.append(current_value)
        
    return pd.Series(portfolio_values, index=prices.index)