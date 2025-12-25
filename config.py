# config.py

# -------------------
# Time settings
# -------------------
HORIZON_DAYS = 5          # 1-week ahead
ROLLING_WINDOW = 60       # already chosen

# -------------------
# Universe
# -------------------
N_STOCKS = 55
N_SECTORS = 11

# -------------------
# Basket construction
# -------------------
LONG_Q = 0.2              # top 20%
SHORT_Q = 0.2             # bottom 20%

# -------------------
# Modeling
# -------------------
RANDOM_SEED = 42
