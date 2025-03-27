# QTradeX Core Framework

Welcome to **QTradeX**, a powerful and flexible Python framework for designing, backtesting, and deploying algorithmic trading bots. Built with cryptocurrency markets in mind, QTradeX provides high-level abstractions for trading strategies, technical indicators, data handling, and optimization, making it an ideal tool for both novice and experienced algo-traders.

This repository contains the core QTradeX framework. 

For a collection of pre-built trading strategies, check out the companion repository: 

[QTradeX Algo Trading Strategies](https://github.com/squidKid-deluxe/qtradex-ai-agents).

---

## Overview

QTradeX simplifies the development of trading bots by offering a modular architecture and robust tools:
- **Core Modules**: Build bots, run backtests, and optimize with a QPSO or LSGA.
- **Data Integration**: Fetch market data from exchanges like BitShares, KuCoin, and others via built-in CCXT integration.
- **Indicators**: Leverage wrapped traditional fininancial technical indicators from Tulip for strategy logic.
- **Visualization**: Plot trades and metrics with ease.
- **Extensibility**: Customize bots and integrate with your preferred exchanges or data sources.

Whether you’re experimenting with moving averages or building complex multi-indicator systems, QTradeX provides the foundation to turn your ideas into actionable trading algorithms.

---

## Features

- **Bot Development**: Subclass `BaseBot` to define custom strategies, indicators, and fitness metrics.
- **Backtesting**: Simulate trades with historical data using `qx.core.backtest`, or use the built-in CLI with `qx.core.dispatch`.
- **Optimization**: Tune parameters automatically with Quantum Particle Swarm Optimization (`qpso.py`) or Local Search Genetic Algorithms (`lsga.py`).
- **Data Pipeline**: Access candle data from any of over a hundred sources.
- **Performance Metrics**: Evaluate bots with built-in fitness functions like ROI, Sortino ratio, and Win Rate with our fitness implementations.
- **DEX and CEX Integration**: Works on both Bitshares Decentralized Exchange and 100+ Centralized Exchanges

## Speed

- **Fully Vectorized**: All indicators are calculated once and cached as vectors
- **Data Cache**: All API candle data from exchanges is collected once and stored on disk
- **Back Propagation**: Achieve 50+ backtests per second on a Raspberry Pi during parameter optimization
- **Instant Backtest**: 1000 candles and 10 indicators in less than a 1/10th of second. 

---

## Roadmap

- **Live Execution**: Finalize and test the live execution pipeline on Centralized Exchange via CCXT SDK and Bitshares Decentralized Exchange
- **Indicators**: Add additional Technical Indicators from sources other than Tulip
- **Tradfi**: Create connector for Traditional Retail Finance Brokerage(s); Stocks, Forex, Comex, etc. 

## Project Structure

```
qtradex/
├── common/           # Utilities like JSON IPC and BitShares nodes
├── core/             # Core bot logic and backtesting
├── indicators/       # Technical indicators and fitness metrics
├── optimizers/       # Optimization algorithms (QPSO, LSGA)
├── plot/             # Visualization tools
├── private/          # Wallet and execution logic
├── public/           # Data fetching and market utilities
└── setup.py          # Installation script
```

Key files:
- `core/base_bot.py`: Base class for creating trading bots.
- `core/dispatch.py`: Allows for easy management of botscripts and tunes.
- `public/data.py`: Manages market data retrieval.
- `indicators/tulipy_wrapped.py`: Wrapped Tulipy indicators for cached speed.

---

## Getting Started

### Installation

```bash
git clone https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK.git QTradeX
cd QTradeX
pip install -e .
```

### Example: Creating a Simple Bot
Here’s a minimal bot using an EMA crossover strategy:

```python
import qtradex as qx
from qtradex.indicators import tulipy as tu
from qtradex.private.signals import Buy, Sell, Thresholds

class EMACrossBot(qx.core.BaseBot):
    def __init__(self):
        self.tune = {"fast_ema": 10, "slow_ema": 50}
    
    def indicators(self, data):
        return {
            "fast_ema": tu.ema(data["close"], self.tune["fast_ema"]),
            "slow_ema": tu.ema(data["close"], self.tune["slow_ema"])
        }
    
    def strategy(self, tick_info, indicators):
        price = tick_info["close"]
        fast = indicators["fast_ema"][-1]
        slow = indicators["slow_ema"][-1]
        if fast > slow:
            return Buy()
        elif fast < slow:
            return Sell()
        return Thresholds(buying=price, selling=price)

# Run the bot
data = qx.public.Data(exchange="kucoin", asset="BTC", currency="USDT", begin="2023-01-01", end="2023-12-31")
wallet = qx.private.PaperWallet({"BTC": 1, "USDT": 0})
bot = EMACrossBot()
qx.core.dispatch(bot, data, wallet)
```

For more examples, see the [QTradeX Algo Trading Strategies](https://github.com/squidKid-deluxe/qtradex-ai-agents) repo.

---

## Usage

1. **Build a Bot**: Subclass `BaseBot` and define your `indicators` and `strategy`.
2. **Backtest**: Use `qx.core.dispatch` with historical data from `qx.Data`.
3. **Optimize**: Run a QPSO of LSGA to fine-tune parameters, which will be stored in `<working directory>/tunes/`.
4. **Deploy**: *Work in progress*

---

## Resources

- **Strategies Repo**: [QTradeX Algo Trading Strategies](https://github.com/squidKid-deluxe/qtradex-ai-agents)
- **Tulipy Docs**: [GitHub](https://tulipindicators.org)
- **CCXT Docs**: [CCXT](https://docs.ccxt.com)

---

## License

This project is licensed WTFPL 

---

Happy coding and trading! Open an issue if you run into problems or have ideas to share.
