import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import existing project modules
from calculate_indicators import calculate_indicators
from get_data import get_stock_data
from fundamentals_screener import fundamentals_screener

# Load environment variables from .env file
load_dotenv()

# Check if API key is available or prompt user to add it
if not os.getenv("GOOGLE_API_KEY"):
    print("Please create a .env file and add your GOOGLE_API_KEY to use the stock insights feature.")
    print("Example: GOOGLE_API_KEY=your_api_key_here")


class StockInsight(BaseModel):
    """Pydantic model for structured stock insights."""
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    confidence_score: float = Field(description="Confidence score (0-100)")
    target_price: float = Field(description="Price target for the stock")
    timeframe: str = Field(description="Recommended holding period (e.g., '2 weeks', '3 months')")
    entry_point: Optional[str] = Field(description="Suggested entry point if recommendation is Buy")
    exit_point: Optional[str] = Field(description="Suggested exit point")
    stop_loss: Optional[float] = Field(description="Recommended stop loss price")
    risk_level: str = Field(description="Risk assessment (Low, Medium, High)")
    reasoning: List[str] = Field(description="Key points explaining the recommendation")
    technical_summary: str = Field(description="Summary of technical indicators")
    fundamental_summary: str = Field(description="Summary of fundamental analysis")
    news_sentiment: str = Field(description="Summary of news sentiment")
    market_context: str = Field(description="Broader market context")


def get_news_articles(ticker: str, company_name: str = None) -> List[Dict[str, str]]:
    """Get recent news articles about the stock."""
    search = DuckDuckGoSearchRun()
    
    # Create search query
    if company_name:
        query = f"{ticker} {company_name} stock news financial analysis recent"
    else:
        query = f"{ticker} stock news financial analysis recent"
    
    try:
        # Get search results
        results = search.run(query)
        
        # Simple parsing of results (assuming a specific format from DuckDuckGo)
        # In a production environment, use a more robust news API
        news_items = []
        for line in results.split("\n"):
            if line.strip():
                news_items.append({"text": line})
                
        return news_items[:5]  # Return top 5 news items
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return [{"text": "Unable to fetch recent news."}]


def get_market_context() -> Dict[str, Any]:
    """Get overall market context using major Indian indices."""
    indices = {
        "^NSEI": "Nifty 50",
        "^BSESN": "Sensex",
        "^NSEBANK": "Nifty Bank"
    }
    
    market_data = {}
    
    try:
        for symbol, name in indices.items():
            data = get_stock_data(symbol, 7, "1d")
            if data is None or len(data) < 6:
                print(f"Skipping {symbol}: No data found or insufficient data.")
                continue
            # Calculate price change and percentage change
            change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
            pct_change = (change / data["Close"].iloc[-2]) * 100
            market_data[name] = {
                "price": round(data["Close"].iloc[-1], 2),
                "change": round(change, 2),
                "percent_change": round(pct_change, 2),
                "trend_5d": "Up" if data["Close"].iloc[-1] > data["Close"].iloc[-5] else "Down"
            }
    except Exception as e:
        print(f"Error fetching market context: {str(e)}")
    
    return market_data


def analyze_stock_sentiment(ticker: str, days: int = 60, interval: str = "1d") -> Dict[str, Any]:
    """
    Analyze a stock using LangChain with Google's Gemini model.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of historical data to analyze
        interval: Data interval ('1d' for daily, '1h' for hourly)
        
    Returns:
        Dictionary with analysis results
    """
    # Ensure we have the API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "error": "Google API key not found. Please set the GOOGLE_API_KEY environment variable."
        }
    
    try:
        # Step 1: Gather all necessary data
        print(f"Fetching data for {ticker}...")
        
        # Get stock price data
        stock_data = get_stock_data(ticker, days, interval)
        if stock_data is None:
            return {"error": f"Could not fetch data for {ticker}"}
        
        # Calculate technical indicators
        technical_data = calculate_indicators(stock_data)
        
        # Get fundamental data
        fundamental_data = fundamentals_screener(ticker)
        
        # Get recent news
        company_name = fundamental_data.get("longName", ticker)
        news_articles = get_news_articles(ticker, company_name)
        
        # Get market context
        market_context = get_market_context()
        
        # Step 2: Prepare input data for LLM in a concise format
        current_price = technical_data["Close"].iloc[-1]
        
        # Extract key technical indicators (most recent values)
        latest_indicators = {
            "price": current_price,
            "rsi": technical_data["RSI"].iloc[-1],
            "macd": technical_data["MACD"].iloc[-1],
            "macd_signal": technical_data["Signal_Line"].iloc[-1],
            "upper_band": technical_data["Upper_Band"].iloc[-1],
            "lower_band": technical_data["Lower_Band"].iloc[-1],
            "vwap": technical_data["VWAP"].iloc[-1],
            "50d_price_change_pct": ((current_price / technical_data["Close"].iloc[-min(50, len(technical_data))]) - 1) * 100 if len(technical_data) >= 50 else 0
        }
        
        # Extract key fundamental data
        key_fundamentals = {
            "current_price": fundamental_data.get("Current Price", "N/A"),
            "target_price": fundamental_data.get("Target Mean Price", "N/A"),
            "pe_ratio": fundamental_data.get("Trailing P/E", "N/A"),
            "forward_pe": fundamental_data.get("Forward P/E", "N/A"),
            "price_to_book": fundamental_data.get("Price to Book", "N/A"),
            "analyst_recommendation": fundamental_data.get("Recommendation", "N/A")
        }
        
        # Step 3: Set up LangChain with Gemini
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.2)
        parser = JsonOutputParser(pydantic_object=StockInsight)
        
        # Step 4: Create prompt template
        template = """
        You are an expert financial analyst with deep knowledge of stock markets, technical analysis, and fundamental analysis. 
        Analyze the following stock data and provide a comprehensive investment recommendation.
        
        Stock Symbol: {ticker}
        Current Date: {current_date}
        
        # Technical Analysis Data:
        Current Price: ${price:.2f}
        RSI (14-period): {rsi:.2f}
        MACD: {macd:.4f}
        MACD Signal Line: {macd_signal:.4f}
        Bollinger Upper Band: ${upper_band:.2f}
        Bollinger Lower Band: ${lower_band:.2f}
        VWAP: ${vwap:.2f}
        50-Day Price Change: {price_change:.2f}%
        
        # Fundamental Data:
        Target Price (Analyst Mean): ${target_price}
        Trailing P/E: {pe_ratio}
        Forward P/E: {forward_pe}
        Price-to-Book: {price_to_book}
        Analyst Recommendation: {analyst_rec}
        
        # Recent News:
        {news}
        
        # Market Context:
        {market_context}
        
        Based on all the above information, provide:
        1. A "Buy", "Sell", or "Hold" recommendation with a confidence score (0-100)
        2. A target price and timeframe for the recommendation
        3. If "Buy", suggest an entry point strategy
        4. Suggest an exit point and stop loss price
        5. Assess the risk level (Low, Medium, High)
        6. Provide key reasoning points for your recommendation
        7. Summarize technical indicators, fundamental factors, news sentiment, and market context
        
        Format your response as a JSON object with the following structure:
        {format_instructions}
        """
        
        # Format the news articles
        news_formatted = "\n".join([f"- {item['text']}" for item in news_articles])
        
        # Format market context
        market_context_formatted = ""
        for index_name, data in market_context.items():
            market_context_formatted += f"{index_name}: ${data['price']} ({data['percent_change']}%, {data['trend_5d']} trend)\n"
        
        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "ticker", "current_date", "price", "rsi", "macd", "macd_signal", 
                "upper_band", "lower_band", "vwap", "price_change",
                "target_price", "pe_ratio", "forward_pe", "price_to_book", "analyst_rec",
                "news", "market_context"
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Step 5: Run the analysis
        chain = prompt | model | parser
        
        result = chain.invoke({
            "ticker": ticker,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "price": latest_indicators["price"],
            "rsi": latest_indicators["rsi"],
            "macd": latest_indicators["macd"],
            "macd_signal": latest_indicators["macd_signal"],
            "upper_band": latest_indicators["upper_band"],
            "lower_band": latest_indicators["lower_band"],
            "vwap": latest_indicators["vwap"],
            "price_change": latest_indicators["50d_price_change_pct"],
            "target_price": key_fundamentals["target_price"],
            "pe_ratio": key_fundamentals["pe_ratio"],
            "forward_pe": key_fundamentals["forward_pe"],
            "price_to_book": key_fundamentals["price_to_book"],
            "analyst_rec": key_fundamentals["analyst_recommendation"],
            "news": news_formatted,
            "market_context": market_context_formatted
        })
        
        return result
    
    except Exception as e:
        import traceback
        print(f"Error analyzing {ticker}: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Analysis failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    insights = analyze_stock_sentiment(ticker, days=60, interval="1d")
    print(json.dumps(insights, indent=2))
