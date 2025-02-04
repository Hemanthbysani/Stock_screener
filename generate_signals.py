from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from calculate_indicators import calculate_indicators

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