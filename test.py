import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Stock_Screener import create_dashboard

# Language: python
import plotly.graph_objects as go

class TestCreateDashboard(unittest.TestCase):
    def setUp(self):
        # Create a dummy datetime index for 10 days
        self.dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        
        # Create dummy stock_data DataFrame with required columns
        data = {
            "Open": np.linspace(100, 110, 10),
            "High": np.linspace(105, 115, 10),
            "Low": np.linspace(95, 105, 10),
            "Close": np.linspace(100, 110, 10),
            "Upper_Band": np.linspace(115, 125, 10),
            "Lower_Band": np.linspace(85, 95, 10)
        }
        self.stock_data = pd.DataFrame(data, index=self.dates)
        self.stock_data.index.name = "Date"
        
        # Create dummy backtest_results dictionary
        self.backtest_results = {
            "buy_dates": [self.dates[1], self.dates[3]],
            "sell_dates": [self.dates[2], self.dates[4]],
            "portfolio_values": list(np.linspace(10000, 11000, 10)),
            "final_value": 11000,
            "total_return": 10.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -5.0,
            "num_trades": 2,
            "current_shares": 10,
            "current_cash": 5000
        }
        self.ticker = "TEST"
    
    def test_create_dashboard_returns_figure(self):
        fig = create_dashboard(self.stock_data, self.backtest_results, self.ticker)
        self.assertIsInstance(fig, go.Figure)
    
    def test_dashboard_title_contains_ticker(self):
        fig = create_dashboard(self.stock_data, self.backtest_results, self.ticker)
        self.assertIn(self.ticker, fig.layout.title.text)
    
    def test_portfolio_value_trace_exists(self):
        fig = create_dashboard(self.stock_data, self.backtest_results, self.ticker)
        trace_names = [trace.name for trace in fig.data if trace.name]
        self.assertIn("Portfolio Value", trace_names)
    
    def test_buy_signal_trace_exists(self):
        # Ensure at least one buy date exists
        if self.backtest_results['buy_dates']:
            fig = create_dashboard(self.stock_data, self.backtest_results, self.ticker)
            trace_names = [trace.name for trace in fig.data if trace.name]
            self.assertIn("Buy Signal", trace_names)
    
    def test_sell_signal_trace_exists(self):
        # Ensure at least one sell date exists
        if self.backtest_results['sell_dates']:
            fig = create_dashboard(self.stock_data, self.backtest_results, self.ticker)
            trace_names = [trace.name for trace in fig.data if trace.name]
            self.assertIn("Sell Signal", trace_names)

if __name__ == '__main__':
    unittest.main()