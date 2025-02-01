import numpy as np
import pandas as pd
import time
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Fetch live stock data using Yahoo Finance
def get_live_data(symbol, num_days=200):
    """Fetch last num_days of stock data using Yahoo Finance."""
    stock = yf.Ticker(symbol + ".NS")  # Append ".NS" for NSE stocks
    df = stock.history(period=f"{num_days}d")

    if df.empty:
        raise ValueError(f"Invalid Stock Symbol: {symbol}. Check if the stock is listed on NSE.")

    df = df[["Close", "High", "Low", "Volume"]]
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)
    
    return df

# Train LSTM Model
def train_lstm(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[["Close", "High", "Low", "Volume"]])

    def create_sequences(data, seq_length=100):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  # Predicting Close Price
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(15),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.column_stack([predicted_price, np.zeros((len(predicted_price), 3))]))[:, 0]
    return model, predicted_price

# Compute Performance Metrics
def compute_metrics(df):
    df["Returns"] = df["Close"].pct_change()
    cumulative_return = (df["Returns"] + 1).cumprod() - 1
    max_drawdown = np.min(cumulative_return - np.maximum.accumulate(cumulative_return))

    buy_trades = df[df["Optimized_Buy"]].shape[0]
    sell_trades = df[df["Optimized_Sell"]].shape[0]
    total_trades = buy_trades + sell_trades
    profitability = (df["Returns"].sum()) * 100  # Convert to %

    return {
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Profitability": round(profitability, 2),
        "Total Trades": total_trades
    }

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Stock Trading Strategy with LSTM & Bollinger Bands", className="text-center"),
    
    dbc.Row([
        dbc.Col(dcc.Input(id="stock-input", type="text", placeholder="Enter Stock Symbol (e.g., RELIANCE)", debounce=True, className="form-control")),
        dbc.Col(dcc.Input(id="days-input", type="number", value=200, min=30, step=10, className="form-control")),
        dbc.Col(dbc.Button("Run Analysis", id="run-btn", color="primary", className="btn-block")),
    ], className="mt-3"),

    dcc.Graph(id="stock-chart", className="mt-4"),
    
    html.Div(id="metrics-output", className="mt-3")
])

@app.callback(
    [Output("stock-chart", "figure"), Output("metrics-output", "children")],
    [Input("run-btn", "n_clicks")],
    [dash.State("stock-input", "value"), dash.State("days-input", "value")]
)
def update_chart(n_clicks, stock, days):
    if not stock:
        return go.Figure(), ""

    df = get_live_data(stock, num_days=days)

    # Compute Bollinger Bands
    df["Middle"] = ta.volatility.bollinger_mavg(df["Close"], window=20)
    df["Upper"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)
    df["Lower"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
    df["Volume_Signal"] = df["Volume"] > df["Volume"].rolling(20).mean()
    df["Buy"] = (df["Close"] < df["Lower"]) & df["Volume_Signal"]
    df["Sell"] = df["Close"] > df["Upper"]

    # Train LSTM
    model, predicted_price = train_lstm(df)
    df.loc[df.index[-len(predicted_price):], "Predicted"] = predicted_price
    df["Optimized_Buy"] = (df["Predicted"] < df["Lower"]) & df["Volume_Signal"]
    df["Optimized_Sell"] = df["Predicted"] > df["Upper"]

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper BB", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower BB", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index[-len(predicted_price):], y=predicted_price, name="LSTM Predicted", line=dict(color="red")))

    fig.add_trace(go.Scatter(x=df[df["Optimized_Buy"]].index, y=df[df["Optimized_Buy"]]["Close"], mode="markers", marker=dict(color="green", size=8), name="Buy"))
    fig.add_trace(go.Scatter(x=df[df["Optimized_Sell"]].index, y=df[df["Optimized_Sell"]]["Close"], mode="markers", marker=dict(color="red", size=8), name="Sell"))

    metrics = compute_metrics(df)
    metrics_output = f"Max Drawdown: {metrics['Max Drawdown']}% | Profitability: {metrics['Profitability']}% | Total Trades: {metrics['Total Trades']}"

    return fig, html.H4(metrics_output)

if __name__ == "__main__":
    app.run_server(debug=True)
