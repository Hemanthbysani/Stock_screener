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
from backtest import backtest
from calculate_indicators import calculate_indicators
from fundamentals_screener import fundamentals_screener
from generate_signals import generate_trading_signals, generate_trading_signals_with_ml
from get_data import get_stock_data, get_nifty_top_10

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            
            signals = generate_trading_signals(stock_data)
            
            # Run backtest
            results = backtest(signals, stock_data, initial_balance, )
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
            
            signals = generate_trading_signals_with_ml(stock_data, ticker)
            # Run backtest
            results = backtest(signals, stock_data, initial_balance, )
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
                signals = generate_trading_signals(stock_data)
                # Run backtest
                results = backtest(signals, stock_data, initial_balance)
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
