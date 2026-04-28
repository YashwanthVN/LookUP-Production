TICKER_MAP = {
    "Automotive": {
        "US": ["TSLA", "F", "GM", "RIVN", "LCID", "NIO"],
        "IN": ["OLAELEC.BO", "TATAMOTORS.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "M&M.NS"]
    },
    "Technology": {
        "US": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "META"],
        "IN": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "LTIM.NS"]
    },
    "Energy": {
        "US": ["XOM", "CVX", "SHEL", "BP"],
        "IN": ["RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS"]
    }
}

def get_peers_statically(ticker):
    """
    Finds the sector for a given ticker and returns its siblings.
    """
    ticker_upper = ticker.upper()
    for sector, regions in TICKER_MAP.items():
        for region, symbols in regions.items():
            if ticker_upper in symbols:
                # Return all other symbols in this sector (both US and IN for global context)
                all_peers = regions["US"] + regions["IN"]
                return [p for p in all_peers if p != ticker_upper][:6]
    return []