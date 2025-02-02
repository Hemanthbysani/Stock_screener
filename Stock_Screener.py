import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

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
        # Pre-calculate common levels:
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
        print(stock_data.head())
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

def generate_trading_signals_with_ml(stock_data):
    """Generate trading signals using a CatBoostClassifier only."""
    try:
        # Ensure indicators are available.
        if 'MACD' not in stock_data.columns or 'RSI' not in stock_data.columns:
            stock_data = calculate_indicators(stock_data)
            if stock_data is None:
                raise ValueError("Error calculating indicators")
        
        # Build feature DataFrame.
        df = pd.DataFrame(index=stock_data.index)
        df['MACD'] = stock_data['MACD']
        df['RSI'] = stock_data['RSI']
        df['Close'] = stock_data['Close']
        df['Upper_Band'] = stock_data['Upper_Band']
        df['Lower_Band'] = stock_data['Lower_Band']
        df['VWAP'] = stock_data['VWAP']
        df['Signal_Line'] = stock_data['Signal_Line']
        df['Fib_0.618'] = stock_data['Fib_0.618']
        df['Fib_0.382'] = stock_data['Fib_0.382']
        
        # Primary signals
        df['macd_buy'] = stock_data['MACD'] > stock_data['Signal_Line']
        df['macd_sell'] = stock_data['MACD'] < stock_data['Signal_Line']
        df['vwap_buy'] = stock_data['Close'] > stock_data['VWAP']
        df['vwap_sell'] = stock_data['Close'] < stock_data['VWAP']
        
        # Secondary signals
        df['rsi_buy'] = stock_data['RSI'] < 30  
        df['rsi_sell'] = stock_data['RSI'] > 70  
        df['bb_buy'] = stock_data['Close'] > stock_data['Lower_Band']
        df['bb_sell'] = stock_data['Close'] < stock_data['Upper_Band']
        df['fib_buy'] = stock_data['Close'] > stock_data['Fib_0.618']
        df['fib_sell'] = stock_data['Close'] < stock_data['Fib_0.382']
        
        # Fill missing values robustly.
        df.fillna(method='ffill', inplace=True)
        df['Upper_Band'] = df['Upper_Band'].fillna(df['Close'] * 1.1)
        df['Lower_Band'] = df['Lower_Band'].fillna(df['Close'] * 0.9)
        
        # Create target variable using technical criteria.
        df['target'] = 0
        buy_cond = (df['macd_buy'] | df['rsi_buy']) & (df['vwap_buy'] | df['bb_buy'] | df['fib_buy'])
        sell_cond = (df['macd_sell'] | df['rsi_sell']) & (df['vwap_sell'] | df['bb_sell'] | df['fib_sell'])
        df.loc[buy_cond, 'target'] = 1
        df.loc[sell_cond, 'target'] = -1
        # Hold condition already set to 0.
        
        # Shift target to avoid lookahead bias.
        df['target'] = df['target'].shift(1).fillna(0)
        
        # Prepare feature matrix X and vector y.
        feature_cols = ['MACD', 'RSI', 'Close', 'Upper_Band', 'Lower_Band', 
                        'VWAP', 'Signal_Line', 'Fib_0.618', 'Fib_0.382',
                        'macd_buy', 'macd_sell', 'vwap_buy', 'vwap_sell', 
                        'rsi_buy', 'rsi_sell', 'bb_buy', 'bb_sell', 'fib_buy', 'fib_sell']
        X = df[feature_cols]
        y = df['target']
        
        # Scale features.
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Chronological Train-Test split.
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train CatBoostClassifier only.
        model = CatBoostClassifier(n_estimators=250, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        
        # Evaluate on the test set.
        y_pred = model.predict(X_test)
        print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))
        
        # Predict signals on all data.
        signals = model.predict(X_scaled).ravel()
        signals_series = pd.Series(signals, index=df.index).fillna(0)
        
        return signals_series
    except Exception as e:
        print(f"Error generating signals with CatBoost: {str(e)}")
        return pd.Series(0, index=stock_data.index)

    
def get_nifty_top_10():
    """Fetch the top 10 NIFTY stocks."""
    # NIFTY 50 tickers (you can use an API or a static list for now)
    nifty_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
                     'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS', 'LT.NS']
    return nifty_tickers

def backtest(stock_data, initial_balance=10000, ml=False, stop_loss_pct=0.1):
    """Improved backtesting function that tracks portfolio value until the latest data point."""
    try:
        cash = initial_balance
        shares = 0
        portfolio_values = []
        buy_dates = []
        sell_dates = []
        stop_loss_price = None
        
        # Generate signals for the entire dataset
        if ml == True:
            signals = generate_trading_signals_with_ml(stock_data)
        else:
            signals = generate_trading_signals(stock_data)
        
        # Iterate through all dates
        for date in stock_data.index:
            current_price = stock_data.loc[date, 'Close']
            signal = signals[date]
            
            # Execute trades based on signals
            if signal == 1 and cash >= current_price:  # Buy signal
                shares_to_buy = int(cash // current_price)  # Ensure whole number of shares
                if shares_to_buy > 0:  # Only record if we actually buy
                    cash -= shares_to_buy * current_price
                    shares += shares_to_buy
                    buy_dates.append(date)
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
        print(f"Error in backtesting: {str(e)}")
        return None

def create_dashboard(stock_data, backtest_results, ticker):
    """Create an interactive dashboard with improved visualizations."""
    try:
        # Update to 2 rows: first for Price & Signals, second for Portfolio Value
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price, Indicators & Trading Signals', 'Portfolio Value')
        )

        # Plot 1: Stock Price & Indicators
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'],
                      line=dict(color='gray', dash='dash'), name='Upper BB'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'],
                      line=dict(color='gray', dash='dash'), name='Lower BB'),
            row=1, col=1
        )
        
        # Overlay Buy/Sell Signals on the first plot
        if backtest_results['buy_dates']:
            fig.add_trace(
                go.Scatter(
                    x=backtest_results['buy_dates'],
                    y=[stock_data.loc[date, 'Close'] for date in backtest_results['buy_dates']],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=15, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        if backtest_results['sell_dates']:
            fig.add_trace(
                go.Scatter(
                    x=backtest_results['sell_dates'],
                    y=[stock_data.loc[date, 'Close'] for date in backtest_results['sell_dates']],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=15, symbol='triangle-down')
                ),
                row=1, col=1
            )

        # Plot 2: Portfolio Value
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=backtest_results['portfolio_values'],
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{ticker} Trading Dashboard',
            height=1500,
            showlegend=True,
            xaxis2_title='Date',
            yaxis_title='Price',
            yaxis2_title='Portfolio Value',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True,
                    bgcolor="#636EFA",
                    thickness=0.05
                ),
                type="date"
            )
        )

        return fig

    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        return go.Figure()

def fundamentals_screener(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        # Fetch additional data from yfinance
        rec = yf_ticker.recommendations
        rec_sum = getattr(yf_ticker, "recommendations_summary", None)
        up_down = getattr(yf_ticker, "upgrades_downgrades", None)
        sustain = yf_ticker.sustainability
        analyst_pt = getattr(yf_ticker, "analyst_price_targets", None)
        earnings_est = getattr(yf_ticker, "earnings_estimate", None)
        revenue_est = getattr(yf_ticker, "revenue_estimate", None)
        earnings_hist = getattr(yf_ticker, "earnings_history", None)
        eps_tr = getattr(yf_ticker, "eps_trend", None)
        eps_rev = getattr(yf_ticker, "eps_revisions", None)
        growth_est = getattr(yf_ticker, "growth_estimates", None)
        funds_data = getattr(yf_ticker, "funds_data", None)
        insider_purch = getattr(yf_ticker, "insider_purchases", None)
        insider_trans = getattr(yf_ticker, "insider_transactions", None)
        insider_roster = getattr(yf_ticker, "insider_roster_holders", None)
        major_hold = yf_ticker.major_holders

        fundamentals = {
            "Current Price": info.get("currentPrice", "N/A"),
            "Target Mean Price": info.get("targetMeanPrice", "N/A"),
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Price to Book": info.get("priceToBook", "N/A"),
            "Total Assets": info.get("totalAssets", "N/A"),
            "Recommendation": info.get("recommendationKey", "N/A"),
            "Earnings Quarterly Growth": info.get("earningsQuarterlyGrowth", "N/A"),
            "recommendations": rec.to_dict('records') if rec is not None else None,
            "recommendations_summary": rec_sum.to_dict('records') if hasattr(rec_sum, 'to_dict') else rec_sum,
            "upgrades_downgrades": up_down.to_dict('records') if hasattr(up_down, 'to_dict') else up_down,
            "sustainability": sustain.to_dict() if hasattr(sustain, 'to_dict') else sustain,
            "analyst_price_targets": analyst_pt,
            "earnings_estimate": earnings_est,
            "revenue_estimate": revenue_est,
            "earnings_history": earnings_hist,
            "eps_trend": eps_tr,
            "eps_revisions": eps_rev,
            "growth_estimates": growth_est,
            "funds_data": str(funds_data),
            "insider_purchases": insider_purch,
            "insider_transactions": insider_trans,
            "insider_roster_holders": insider_roster,
            "major_holders": major_hold
        }

        # Convert nested dictionaries or lists to pretty JSON strings
        for key, value in fundamentals.items():
            if isinstance(value, (dict, list)):
                fundamentals[key] = json.dumps(value, indent=2)
                    
        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}
    
# Layout remains the same as before
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stock Trading Backtest Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Ticker Symbol"),
                    dcc.Input(
                        id='ticker-input',
                        value='INFY',
                        type='text',
                        className="form-control mb-3"
                    ),
                    html.Label("Number of Days"),
                    dcc.Input(
                        id='days-input',
                        value=365,
                        type='number',
                        className="form-control mb-3"
                    ),
                    html.Label("Timeframe"),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '1 Hour', 'value': '1h'},
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '1 Week', 'value': '1wk'}
                        ],
                        value='1d',
                        className="form-control mb-3"
                    ),
                    html.Label("Initial Balance"),
                    dcc.Input(
                        id='balance-input',
                        value=10000,
                        type='number',
                        className="form-control mb-3"
                    ),
                    dbc.Button("Run Backtest", id='run-btn', color='primary', className="mr-2"),
                    dbc.Button("Run ML Backtest", id='ml-btn', color='secondary', className="mr-2"),
                    dbc.Button("Run NIFTY Backtest", id='nifty-btn', color='info'),
                    dbc.Button("Fetch Fundamentals", id='fundamentals-btn', color='info', className="mr-2"),
                    html.Div(id='error-message', className="text-danger mt-3")
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    dcc.Graph(id='backtest-graph'),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Backtest Results", className="card-title"),
                            html.Pre(id='metrics', className="mb-0")
                        ])
                    ])
                ]
            )
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.H3("Fundamentals Screener", className="text-center mt-4"),
            html.Div(id='fundamentals-div', className="mt-3", style={"border": "1px solid #ccc", "padding": "10px"})
        ], width=12)
    ])
], fluid=True)


@app.callback(
    [Output('backtest-graph', 'figure'),
     Output('metrics', 'children'),
     Output('error-message', 'children')],
    [Input('run-btn', 'n_clicks'),
     Input('ml-btn', 'n_clicks'),
     Input('nifty-btn', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('days-input', 'value'),
     State('timeframe-dropdown', 'value'),
     State('balance-input', 'value')]
)
def update_dashboard(n_clicks, ml_clicks, nifty_clicks, ticker, days, interval, initial_balance):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "", ""
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if triggered_id == 'run-btn':  # Backtest for a single stock
            # Validate inputs
            if not ticker or not days or not initial_balance:
                raise ValueError("Please fill in all fields")
            
            # Get stock data
            stock_data = get_stock_data(ticker, days, interval)
            if stock_data is None:
                raise ValueError(f"Could not fetch data for {ticker}")
            
            # Calculate indicators
            stock_data = calculate_indicators(stock_data)
            if stock_data is None:
                raise ValueError("Error calculating indicators")
            
            # Run backtest
            results = backtest(stock_data, initial_balance)
            if results is None:
                raise ValueError("Error running backtest")
            
            fig = create_dashboard(stock_data, results, ticker)
            
            # Format metrics
            metrics_text = f"""
            Final Portfolio Value: INR{results['final_value']:,.2f}
            Total Return: {results['total_return']:.2f}%
            Sharpe Ratio: {results['sharpe_ratio']:.2f}
            Maximum Drawdown: {results['max_drawdown']:.2f}%
            Number of Trades: {results['num_trades']}
            """
            
            return fig, metrics_text, ""
        
        elif triggered_id == 'ml-btn':  # Backtest with ML model
            # Validate inputs
            if not ticker or not days or not initial_balance:
                raise ValueError("Please fill in all fields")
            
            # Get stock data
            stock_data = get_stock_data(ticker, days, interval)
            if stock_data is None:
                raise ValueError(f"Could not fetch data for {ticker}")
            
            # Calculate indicators
            stock_data = calculate_indicators(stock_data)
            if stock_data is None:
                raise ValueError("Error calculating indicators")
            
            # Run backtest
            results = backtest(stock_data, initial_balance, ml=True)
            if results is None:
                raise ValueError("Error running backtest")
            
            fig = create_dashboard(stock_data, results, ticker)
            
            # Format metrics
            metrics_text = f"""
            Final Portfolio Value: INR{results['final_value']:,.2f}
            Total Return: {results['total_return']:.2f}%
            Sharpe Ratio: {results['sharpe_ratio']:.2f}
            Maximum Drawdown: {results['max_drawdown']:.2f}%
            Number of Trades: {results['num_trades']}
            """
            
            return fig, metrics_text, ""
        
        elif triggered_id == 'nifty-btn':  # Backtest for NIFTY top 10 stocks
            # Get NIFTY top 10 stocks
            nifty_tickers = get_nifty_top_10()
            
            all_metrics = []
            for ticker in nifty_tickers:
                # Get stock data
                stock_data = get_stock_data(ticker, days, interval)
                if stock_data is None:
                    continue
                
                # Calculate indicators
                stock_data = calculate_indicators(stock_data)
                if stock_data is None:
                    continue
                
                # Run backtest
                results = backtest(stock_data, initial_balance)
                if results is None:
                    continue
                
                # Collect metrics
                metrics_text = f"""
                {ticker}:
                Final Portfolio Value: INR{results['final_value']:,.2f}
                Total Return: {results['total_return']:.2f}%
                Sharpe Ratio: {results['sharpe_ratio']:.2f}
                Maximum Drawdown: {results['max_drawdown']:.2f}%
                Number of Trades: {results['num_trades']}
                """
                all_metrics.append(metrics_text)
            
            fig = create_dashboard(stock_data, results, ticker)
            return fig, "\n\n".join(all_metrics), ""
    
    except Exception as e:
        return "", "", str(e)

@app.callback(
    Output('fundamentals-div', 'children'),
    Input('fundamentals-btn', 'n_clicks'),
    State('ticker-input', 'value')
)
def update_fundamentals(n_clicks, ticker):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'fundamentals-btn':
        fundamentals = fundamentals_screener(ticker)
        if not fundamentals:
            return html.P("No fundamental data available.")
        content = []
        for key, value in fundamentals.items():
            content.append(html.P(f"{key}: {value}"))
        return html.Div(content)

if __name__ == '__main__':
    app.run_server(debug=True)
