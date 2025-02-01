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
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def calculate_indicators(df):
    """Calculate technical indicators with error handling."""
    try:
        # Create a copy to avoid modifying the original dataframe
        stock_data = df.copy()
        
        # MACD
        stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        stock_data['SMA'] = stock_data['Close'].rolling(window=20, min_periods = 2).mean()
        stock_data['std_dev'] = stock_data['Close'].rolling(window=20, min_periods = 2).std()
        stock_data['Upper_Band'] = stock_data['SMA'] + (stock_data['std_dev'] * 2)
        stock_data['Lower_Band'] = stock_data['SMA'] - (stock_data['std_dev'] * 2)
        
        # RSI
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        stock_data['MACD'] = stock_data['MACD'].fillna(0)  # Neutral MACD value
        stock_data['RSI'] = stock_data['RSI'].fillna(0)  # Neutral RSI value
        stock_data['std_dev'] = stock_data['std_dev'].fillna(0)
        stock_data['SMA'] = stock_data['SMA'].fillna(0)  # Neutral std_dev value
        stock_data['Close'] = stock_data['Close'].fillna(method='ffill')  # Forward fill for Close price
        stock_data['Upper_Band'] = stock_data['Upper_Band'].fillna(stock_data['Close'] * 1.1)  # Default to 10% above Close
        stock_data['Lower_Band'] = stock_data['Lower_Band'].fillna(stock_data['Close'] * 0.9)  # Default to 10% below Close
        # stock_data['target'] = stock_data['target'].fillna(0)  # Default to Hold # Forward fill for VIX

        return stock_data
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return None

def get_stock_data(ticker, days, interval):
    """Get stock data with proper error handling and validation."""
    try:
        # Calculate the start date based on the number of days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, multi_level_index=False)
        
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
            
        # Ensure the index is datetime
        stock_data.index = pd.to_datetime(stock_data.index)
        
        # Handle gaps in the data by forward-filling missing values
        stock_data = stock_data.asfreq('H' if interval == '1h' else 'D' if interval == '1d' else 'W').fillna(method='ffill')
        
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def generate_trading_signals(stock_data):
    """Generate trading signals based on technical indicators."""
    try:
        # Make sure all Series are aligned by using the same DataFrame
        df = pd.DataFrame(index=stock_data.index)
        
        # Create individual conditions and store them in the DataFrame
        df['macd_condition'] = stock_data['MACD'] > stock_data['Signal_Line']
        df['price_condition_buy'] = stock_data['Close'] > stock_data['Lower_Band']
        df['rsi_condition_buy'] = stock_data['RSI'] < 60
        
        df['macd_condition_sell'] = stock_data['MACD'] < stock_data['Signal_Line']
        df['price_condition_sell'] = stock_data['Close'] < stock_data['Upper_Band']
        df['rsi_condition_sell'] = stock_data['RSI'] > 40
        
        # Initialize signals series
        signals = pd.Series(0, index=stock_data.index)
        
        # Buy signals (all conditions must be True)
        buy_signals = (
            df['macd_condition'] & 
            df['price_condition_buy'] & 
            df['rsi_condition_buy']
        )
        
        # Sell signals (all conditions must be True)
        sell_signals = (
            df['macd_condition_sell'] & 
            df['price_condition_sell'] & 
            df['rsi_condition_sell']
        )
        # Set signals
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        # Fill NaN values with 0
        signals = signals.fillna(0)
        return signals
        
    except Exception as e:
        print(f"Error generating signals: {str(e)}")
        return pd.Series(0, index=stock_data.index)
    
def generate_trading_signals_with_ml(stock_data):
    """Generate trading signals based on machine learning model."""
    try:
        # Make sure all Series are aligned by using the same DataFrame
        df = pd.DataFrame(index=stock_data.index)
        
        # Calculate indicators if not already present
        if 'MACD' not in stock_data.columns:
            stock_data = calculate_indicators(stock_data)
            if stock_data is None:
                raise ValueError("Error calculating indicators")
        
        # Create feature columns
        df['MACD'] = stock_data['MACD']
        df['RSI'] = stock_data['RSI']
        df['Close'] = stock_data['Close']
        df['Upper_Band'] = stock_data['Upper_Band']
        df['Lower_Band'] = stock_data['Lower_Band']
        
        #  # Fill NaN values with appropriate defaults
        # df['MACD'] = df['MACD'].fillna(df.mean())  # Neutral MACD value
        # df['RSI'] = df['RSI'].fillna(method=df.mean())  # Neutral RSI value
        # df['Close'] = df['Close'].fillna(method='ffill')  # Forward fill for Close price
        # df['Upper_Band'] = df['Upper_Band'].fillna(df['Close'] * 1.1)  # Default to 10% above Close
        # df['Lower_Band'] = df['Lower_Band'].fillna(df['Close'] * 0.9)  # Default to 10% below Close
        df['target'] = 0  # Default to Hold # Forward fill for VIX
        
        print(df.head())
        # Buy condition: MACD > Signal line, RSI < 60, Close > Lower Band
        df.loc[
            (stock_data['MACD'] > stock_data['Signal_Line']) & 
            (stock_data['RSI'] < 60) & 
            (stock_data['Close'] > stock_data['Lower_Band']), 'target'] = 1

        # Sell condition: MACD < Signal line, RSI > 40, Close < Upper Band
        df.loc[
            (stock_data['MACD'] < stock_data['Signal_Line']) & 
            (stock_data['RSI'] > 40) & 
            (stock_data['Close'] < stock_data['Upper_Band']), 'target'] = -1
        df['target'] = df['target'].shift(1).fillna(0) # Shift to avoid lookahead bias
        # Shift target to avoid lookahead bias (use previous day's target for today's decision)
        
       
        
        # Split the data into features (X) and target (y)
        X = df[['MACD', 'RSI', 'Close', 'Upper_Band', 'Lower_Band']]
        y = df['target']
        # print("X+y=" + df.head())
        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        estimators = [
                    ('rf', RandomForestClassifier(n_estimators=150, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42),
                     'ab', AdaBoostClassifier(n_estimators=150, random_state=42))]
        # Initialize and train the model (using RandomForestClassifier as an example)
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        model.fit(X_train, y_train)
        if not hasattr(model, 'estimators_'):
            raise ValueError("Model training failed")
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Predict the trading signals on the entire dataset
        signals = model.predict(X)
        
        # Check if signals are valid
        if signals is None or len(signals) == 0:
            raise ValueError("Invalid signals generated by the model")
        
        # Map the predictions back to the original index
        signals_series = pd.Series(signals, index=df.index)
        
        # Fill NaN values with 0
        signals_series = signals_series.fillna(0)
        
        return signals_series
    except Exception as e:
        print(f"Error generating signals with ML: {str(e)}")
        return pd.Series(0, index=stock_data.index)

    
def get_nifty_top_10():
    """Fetch the top 10 NIFTY stocks."""
    # NIFTY 50 tickers (you can use an API or a static list for now)
    nifty_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
                     'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS', 'LT.NS']
    return nifty_tickers


def backtest(stock_data, initial_balance=10000, ml=False):
    """Improved backtesting function that tracks portfolio value until the latest data point."""
    try:
        cash = initial_balance
        shares = 0
        portfolio_values = []
        buy_dates = []
        sell_dates = []
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
            elif signal == -1 and shares > 0:  # Sell signal
                cash += shares * current_price
                shares = 0
                sell_dates.append(date)
            
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
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price & Indicators', 'Portfolio Value', 'Trade Signals')
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

        # Plot 3: Buy/Sell Signals
        if backtest_results['buy_dates']:
            fig.add_trace(
                go.Scatter(
                    x=backtest_results['buy_dates'],
                    y=[stock_data.loc[date, 'Close'] for date in backtest_results['buy_dates']],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=3, col=1
            )
        
        if backtest_results['sell_dates']:
            fig.add_trace(
                go.Scatter(
                    x=backtest_results['sell_dates'],
                    y=[stock_data.loc[date, 'Close'] for date in backtest_results['sell_dates']],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'{ticker} Trading Dashboard',
            height=2500,
            showlegend=True,
            xaxis3_title='Date',
            yaxis_title='Price',
            yaxis2_title='Portfolio Value',
            yaxis3_title='Price',
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
                        value='1h',
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


if __name__ == '__main__':
    app.run_server(debug=True)