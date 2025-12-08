Draft Project Write-Up — TSFL Factors (Current Stage)
2. Construction of TSFL Factor Returns

In the first stage of the project, we replicate the Time-Series Factor Library (TSFL) methodology using publicly available data from Yahoo Finance. The objective is to generate a set of daily factor return series that can serve as explanatory variables for later machine-learning models of stock return predictability.

2.1 Data Sources and Universe

For each TSFL factor, we identify a liquid, tradeable Yahoo Finance proxy—typically a broad ETF or index tracker—that captures the intended systematic exposure (e.g., equity market, credit spreads, commodities, FX, inflation, value, momentum, small-cap).
with equivalent Yahoo tickers, we ensure reproducibility while preserving the behavioral properties of the original TSFL specification.

Daily adjusted closing prices are downloaded using the yfinance library for a history of approximately ten years (2010–2025). Adjusted prices allow us to work directly with total-return series without manually accounting for dividend effects or splits.

2.2 Return Computation and Cleaning

From these price series, we compute simple daily returns:

𝑟
𝑡
=
𝑃
𝑡
−
𝑃
𝑡
−
1
𝑃
𝑡
−
1
.
r
t
	​

=
P
t−1
	​

P
t
	​

−P
t−1
	​

	​

.

Simple returns are chosen rather than log returns to maintain consistency with the classical factor-modeling literature and to ensure interpretability in the upcoming beta estimations.

A lightweight cleaning procedure is applied:

rows where all factor returns are missing are removed (market holidays),

partial rows are kept to avoid excessive data loss,

no aggressive filtering is applied at this stage to retain the natural volatility of the factor series.

This results in a clean, aligned panel of daily TSFL factor returns, typically covering more than 3,000 trading days.

2.3 Interpretation of TSFL Factors

Each TSFL factor is defined to isolate a particular dimension of market risk. For example:

TSFL Equity US captures broad equity-market directionality,

TSFL Credit proxies global credit-spread movements,

TSFL Momentum reflects cross-sectional momentum behavior,

TSFL Small Cap measures relative small-cap vs large-cap performance,

TSFL FX / Commodities / Inflation capture macroeconomic exposures.

These factors form the core explanatory variables that later models (OLS, Ridge, Random Forest, MLP, etc.) will use to describe or predict individual stock movements.