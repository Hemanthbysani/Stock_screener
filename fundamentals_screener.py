import yfinance as yf
import json
def fundamentals_screener(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        print("Ticker:", ticker)
        info = yf_ticker.info
        print("Info:", info)
        rec = yf_ticker.recommendations
        print("Recommendations:", rec)
        # Get all the attributes safely with default value of "Not found"
        try:
            rec_sum = getattr(yf_ticker, "recommendations_summary", "Not found")
        except:
            rec_sum = "Not found"
        try:
            sustain = getattr(yf_ticker, "sustainability", "Not found")
        except:
            sustain = "Not found"
        try:
            up_down = getattr(yf_ticker, "upgrades_downgrades", "Not found")
        except:
            up_down = "Not found"
        try:
            analyst_pt = getattr(yf_ticker, "analyst_price_targets", "Not found")
        except:
            analyst_pt = "Not found"
        try:
            earnings_est = getattr(yf_ticker, "earnings_estimate", "Not found")
        except:
            earnings_est = "Not found"
        try:
            revenue_est = getattr(yf_ticker, "revenue_estimate", "Not found")
        except:
            revenue_est = "Not found"
        try:
            earnings_hist = getattr(yf_ticker, "earnings_history", "Not found")
        except:
            earnings_hist = "Not found"
        try:
            eps_tr = getattr(yf_ticker, "eps_trend", "Not found")
        except:
            eps_tr = "Not found"
        try:
            eps_rev = getattr(yf_ticker, "eps_revisions", "Not found")
        except:
            eps_rev = "Not found"
        try:
            growth_est = getattr(yf_ticker, "growth_estimates", "Not found")
        except:
            growth_est = "Not found"
        try:
            funds_data = getattr(yf_ticker, "funds_data", "Not found")
        except:
            funds_data = "Not found"
        try:
            insider_purch = getattr(yf_ticker, "insider_purchases", "Not found")
        except:
            insider_purch = "Not found"
        try:
            insider_trans = getattr(yf_ticker, "insider_transactions", "Not found")
        except:
            insider_trans = "Not found"
        try:
            insider_roster = getattr(yf_ticker, "insider_roster_holders", "Not found")
        except:
            insider_roster = "Not found"
        try:
            major_hold = yf_ticker.major_holders
        except:
            major_hold = "Not found"

        fundamentals = {
            "Current Price": info.get("currentPrice", "Not found"),
            "Target Mean Price": info.get("targetMeanPrice", "Not found"),
            "Trailing P/E": info.get("trailingPE", "Not found"),
            "Forward P/E": info.get("forwardPE", "Not found"),
            "Price to Book": info.get("priceToBook", "Not found"),
            "Total Assets": info.get("totalAssets", "Not found"),
            "Recommendation": info.get("recommendationKey", "Not found"),
            "Earnings Quarterly Growth": info.get("earningsQuarterlyGrowth", "Not found"),
            "recommendations": rec.to_dict('records') if rec is not None else "Not found",
            "recommendations_summary": rec_sum.to_dict('records') if hasattr(rec_sum, 'to_dict') else "Not found" if rec_sum is None else rec_sum,
            "upgrades_downgrades": up_down.to_dict('records') if hasattr(up_down, 'to_dict') else "Not found" if up_down is None else up_down,
            "sustainability": sustain.to_dict() if hasattr(sustain, 'to_dict') else "Not found" if sustain is None else sustain,
            "analyst_price_targets": "Not found" if analyst_pt is None else analyst_pt,
            "earnings_estimate": "Not found" if earnings_est is None else earnings_est,
            "revenue_estimate": "Not found" if revenue_est is None else revenue_est,
            "earnings_history": "Not found" if earnings_hist is None else earnings_hist,
            "eps_trend": "Not found" if eps_tr is None else eps_tr,
            "eps_revisions": "Not found" if eps_rev is None else eps_rev,
            "growth_estimates": "Not found" if growth_est is None else growth_est,
            "funds_data": "Not found" if funds_data is None else str(funds_data),
            "insider_purchases": "Not found" if insider_purch is None else insider_purch,
            "insider_transactions": "Not found" if insider_trans is None else insider_trans,
            "insider_roster_holders": "Not found" if insider_roster is None else insider_roster,
            "major_holders": "Not found" if major_hold is None else major_hold
        }

        # Convert nested dictionaries or lists to pretty JSON strings
        for key, value in fundamentals.items():
            if isinstance(value, (dict, list)):
                fundamentals[key] = json.dumps(value, indent=2)
                    
        return fundamentals
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return fundamentals
    finally:
        print("Fundamentals fetched successfully.")
