import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

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
    
def get_nifty_top_10():
    """Fetch the top 10 NIFTY stocks."""
    # NIFTY 50 tickers (you can use an API or a static list for now)
    nifty_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
                     'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS', 'LT.NS']
    return nifty_tickers