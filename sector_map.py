import pandas as pd

sector_map = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials",

    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "UNP": "Industrials", "HON": "Industrials",

    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "MRK": "Healthcare", "ABT": "Healthcare",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "OXY": "Energy", "SLB": "Energy",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",

    # Consumer Staples
    "KO": "Consumer Staples", "WMT": "Consumer Staples",
    "COST": "Consumer Staples", "PG": "Consumer Staples",
    "PEP": "Consumer Staples",

    # Utilities
    "DUK": "Utilities", "NEE": "Utilities", "D": "Utilities",
    "SO": "Utilities", "AEP": "Utilities",

    # Materials
    "LIN": "Materials", "FCX": "Materials", "NEM": "Materials",
    "DD": "Materials", "VMC": "Materials",

    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services",

    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate",
    "SPG": "Real Estate", "EQIX": "Real Estate", "PSA": "Real Estate",
}

df_sector = (
    pd.DataFrame(sector_map.items(), columns=["stock", "sector"])
    .sort_values("sector")
)

df_sector.to_csv("data/sector_map.csv", index=False)

print("sector_map.csv written successfully")
