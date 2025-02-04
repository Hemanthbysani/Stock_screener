
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