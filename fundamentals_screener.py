import yfinance as yf
import json
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