# Stock Screener & Trading Strategy Analyzer

## Description
A comprehensive tool for analyzing stocks, generating trading signals using multiple strategies, and backtesting performance. This project combines traditional technical analysis, machine learning, and reinforcement learning approaches to provide actionable trading insights.

## Features
### Multiple Signal Generation Methods:
- Traditional technical indicators (MACD, RSI, Bollinger Bands, VWAP, Fibonacci)
- Machine learning-based prediction
- Reinforcement learning trading agent

### Interactive Dashboard:
- Real-time stock data visualization
- Technical indicator overlays
- Trading signal markers
- Performance metrics

### Backtest Engine:
- Simulate trading strategies
- Calculate key performance metrics
- Compare different approaches

### Custom Technical Indicators:
- Moving averages
- MACD
- RSI
- Bollinger Bands
- VWAP
- Fibonacci retracements

## Installation

### Requirements
Ensure you have Python 3.8+ installed. Install the necessary dependencies using:
```sh
pip install pandas numpy yfinance plotly dash dash-bootstrap-components scikit-learn gym stable-baselines3 matplotlib
```

## Usage
To start the dashboard application, run:
```sh
python Stock_Screener.py
```
Navigate to [http://127.0.0.1:8050/](http://127.0.0.1:8050/) in your web browser to access the dashboard.

## Project Structure
```
├── Stock_Screener.py          # Main application with Dash dashboard
├── calculate_indicators.py    # Technical indicator calculation functions
├── generate_signals.py        # Trading signal generation strategies
├── backtest.py                # Backtesting engine for strategy evaluation
├── utils.py                   # Utility functions shared across modules
├── models/                    # Directory for saved ML/RL models
```

## Trading Strategies
### Traditional Technical Analysis
Combines several technical indicators to generate buy/sell signals:
- MACD crossovers
- RSI overbought/oversold levels
- Bollinger Band breakouts
- VWAP price comparisons
- Fibonacci retracement levels

### Machine Learning
Uses Random Forest classification to predict buy/sell signals based on historical patterns in technical indicators.

### Reinforcement Learning
Implements a Deep Q-Network (DQN) agent that learns optimal trading strategies by interacting with a simulated trading environment.

## Configuration
Customize parameters in the dashboard interface:
- Ticker symbol
- Lookback period
- Trading strategy
- Initial investment amount

## Contributing
1. Fork the repository
2. Create your feature branch:
   ```sh
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes:
   ```sh
   git commit -m 'Add some amazing feature'
   ```
4. Push to the branch:
   ```sh
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

## License
This project is licensed under the MIT License.

## Acknowledgements
- **Yahoo Finance API** for providing stock data
- **Dash** for the interactive web application
- **scikit-learn** for machine learning components
- **Stable-Baselines3** for reinforcement learning implementation
