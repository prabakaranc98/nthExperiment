# Quant Research

**Type:** Lens + Craft
**Serves:** The intersection of mathematical rigor, market understanding, and statistical inference at scale. Quant research sharpens the empirical and statistical instincts that ML research often dulls — because in markets, you can't overfit to the past and survive.

---

## Why This

Markets are the hardest prediction problem humans have designed: adversarial, non-stationary, with feedback loops that destroy the signal once it's found. Quant research forces a different kind of statistical discipline than academic ML — you have to think about costs, turnover, regime change, execution, and the fundamental impossibility of backtesting your way to ground truth.

The cross-pollination goes both ways:
- Quant methods (time-series modeling, risk decomposition, factor models) are directly applicable to ML systems
- ML methods (deep learning, NLP on alternative data, RL for execution) are reshaping quantitative finance
- The epistemological discipline of quant — "is this result real or am I fooling myself?" — is exactly what ML research needs more of

---

## Core Areas

### Financial Time Series and Stochastic Processes
Markets as stochastic processes. Brownian motion, geometric Brownian motion, mean reversion, jump processes. The basis for everything in quantitative finance.

**Build:** Implement a GBM simulator. Fit it to real price data. Identify where it fails.

### Factor Models and Risk Decomposition
How do you explain returns? CAPM is a one-factor model. Fama-French is three factors. Modern quant uses hundreds of factors. The framework: alpha (skill) vs. beta (market exposure), risk decomposition, portfolio construction under constraints.

**Build:** Implement a simple three-factor model regression on a small portfolio. Decompose returns into factor and idiosyncratic components.

### Statistical Arbitrage and Pairs Trading
The basic quant strategy: find two assets whose prices move together, wait for divergence, bet on convergence. Requires cointegration testing, Kalman filters, and discipline about when the relationship breaks.

**Build:** Implement a Kalman filter-based pairs trading strategy on two correlated ETFs. Backtest honestly — including transaction costs.

### Backtesting Discipline
The cardinal sin of quant research is an overfit backtest. How do you tell if a strategy is real? Walk-forward testing, out-of-sample testing, the multiple comparisons problem in strategy search.

**The most important question:** If a strategy works in backtest, what is the probability it works live? How do you estimate this?

### ML for Finance
Where does ML actually add value in finance (and where does it destroy it via overfitting)?
- NLP on alternative data: earnings call transcripts, news, filings
- Deep learning for volatility forecasting
- RL for execution: optimal order splitting
- Graph neural networks for market microstructure

---

## Build Progression

- [ ] **Step 1:** Implement geometric Brownian motion simulation. Fit to one equity. Quantify where the model fails (fat tails, volatility clustering).

- [ ] **Step 2:** Implement Fama-French 3-factor regression from scratch. Use public data (Ken French's data library). Decompose returns for a small portfolio.

- [ ] **Step 3:** Implement a Kalman filter pairs trading strategy. Backtest *honestly*: subtract 10bps per trade, use realistic slippage, test out-of-sample.

- [ ] **Step 4:** Read *Advances in Financial Machine Learning* Ch. 7 (cross-validation in finance). Apply combinatorial purged cross-validation to your strategy. Does it still work?

- [ ] **Step 5:** Write: "Why backtesting is harder than you think" — a technical note connecting quant discipline to ML evaluation methodology.

---

## Essential Resources

- [ ] **Advances in Financial Machine Learning — Marcos López de Prado**
  - Ch. 1–2: Financial data structures
  - Ch. 7: Cross-validation in finance *(the most important chapter)*
  - Ch. 16: ML asset allocation

- [ ] **Quantitative Finance — Paul Wilmott** *(selected)*
  - Part 1: Foundations of stochastic calculus

- [ ] **Statistical Arbitrage — Andrew Pole** *(selected)*
  - Ch. 2–4: Pairs trading mechanics and cointegration

- [ ] **Active Portfolio Management — Grinold & Kahn** *(selected)*
  - Ch. 1–3: The fundamental law of active management

- [ ] **Stochastic Calculus for Finance I — Shreve**
  - Ch. 1–3: Binomial model, martingales, risk-neutral pricing

---

## Build Log

| # | Strategy / Model Built | Sharpe (IS) | Sharpe (OOS) | Honest? | Date |
|---|----------------------|-------------|--------------|---------|------|
| — | — | — | — | — | — |
