import numpy as np
import pandas as pd
from utils import calculate_portfolio_value


def backtest(signal_data, stock_data, initial_balance=10000, stop_loss_pct=0.1):
    """Improved backtesting function that tracks portfolio value until the latest data point."""
    try:
        # Reset all tracking variables
        cash = float(initial_balance)  # Ensure float type
        shares = 0
        portfolio_values = []
        buy_dates = []
        sell_dates = []
        stop_loss_price = None
        signals = signal_data.copy()  # Create a copy to prevent modifications to original data

        # Iterate through all dates
        for date in stock_data.index:
            current_price = stock_data.loc[date, 'Close']
            signal = signals[date]
            # Execute trades based on signals
            if signal == 1 and cash >= current_price and shares == 0:  # Buy signal only if no position exists
                shares_to_buy = int(cash // current_price)  # Ensure whole number of shares
                if shares_to_buy > 0:
                    cash -= shares_to_buy * current_price * 1.0003 # For brokerage
                    shares += shares_to_buy
                    buy_dates.append(date)
                    stop_loss_price = current_price * (1 - stop_loss_pct)  # Set stop-loss price
                    stop_loss_price = current_price * (1 - stop_loss_pct)  # Set stop-loss price
            elif signal == -1 and shares > 0:  # Sell signal
                cash += shares * current_price
                shares = 0
                sell_dates.append(date)
                stop_loss_price = None  # Reset stop-loss price

            # Check for stop-loss condition
            if stop_loss_price is not None and shares > 0 and current_price <= stop_loss_price:
                cash += shares * current_price
                shares = 0
                sell_dates.append(date)
                stop_loss_price = None  # Reset stop-loss price

            # Calculate current portfolio value (cash + value of held shares)
            current_portfolio_value = cash + (shares * current_price)
            portfolio_values.append(current_portfolio_value)

        # Ensure we have a portfolio value for every date
        if len(portfolio_values) < len(stock_data):
            missing_dates = len(stock_data) - len(portfolio_values)
            last_value = portfolio_values[-1] if portfolio_values else initial_balance
            portfolio_values.extend([last_value] * missing_dates)

        # Calculate final position value
        final_value = portfolio_values[-1]

        # Calculate returns and metrics
        returns = np.array(portfolio_values) / initial_balance - 1
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak

        metrics = {
            'final_value': final_value,
            'total_return': ((final_value - initial_balance) / initial_balance) * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0,
            'max_drawdown': drawdown.min() * 100 if len(drawdown) > 0 else 0,
            'num_trades': len(buy_dates),
            'current_shares': shares,
            'current_cash': cash,
            'portfolio_values': portfolio_values,
            'buy_dates': buy_dates,
            'sell_dates': sell_dates
        }
        return metrics
    except Exception as e:
        print(f"Error generating backtest: {str(e)}")
        return pd.Series(0, index=stock_data.index)