import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from plotly.subplots import make_subplots
from calculate_indicators import calculate_indicators
from get_data import get_stock_data, get_nifty_top_10
from generate_signals import generate_trading_signals, generate_trading_signals_with_ml

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
