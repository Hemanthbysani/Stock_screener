import yfinance as yf
import pandas as pd
import json
def get_data(ticker):
    return

if __name__ == '__main__': 
    data = get_data('AAPL')
    ticker = yf.Ticker('AAPL')
    data: dict = ticker.get_analyst_price_targets()
    print("Current Price: "+str(data["current"]))
    print("High Price: "+str(data["high"]))
    print("Low Price: "+str(data["low"]))
    print("Mean Price: "+str(data["mean"]))
    print("Median Price: "+str(data["median"]))
    print("75th percentile: " + str((data["mean"]+data["low"])/2))
