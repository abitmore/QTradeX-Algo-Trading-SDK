# 🚀 QTradeX Core — Build, Backtest & Optimize AI-Powered Crypto Trading Bots

<p align="center">
  <img src="screenshots/Screenshot from 2025-05-02 18-50-54.png" width="100%" alt="QTradeX Demo Screenshot">
</p>

> 📸 See [screenshots.md](screenshots.md) for more visuals  
> 📚 Read the core docs on [QTradeX SDK DeepWiki](https://deepwiki.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK)  
> 🤖 Explore the bots at [QTradeX AI Agents DeepWiki](https://deepwiki.com/squidKid-deluxe/QTradeX-AI-Agents)  
> 💬 Join our [Telegram Group](https://t.me/qtradex_sdk) for discussion & support

---

## ⚡️ TL;DR
**QTradeX** is a lightning-fast Python framework for designing, backtesting, and deploying algorithmic trading bots, built for **crypto markets** with support for **100+ exchanges**, **AI-driven optimization**, and **blazing-fast vectorized execution**.

---

## 🎯 Why QTradeX?

Whether you're exploring a simple EMA crossover or engineering a strategy with 20+ indicators and genetic optimization, QTradeX gives you:

✅ Modular Architecture  
✅ Tulip + CCXT Integration  
✅ Custom Bot Classes  
✅ Fast, Disk-Cached Market Data  
✅ Near-Instant Backtests (even on Raspberry Pi!)

---

## 🔍 Features at a Glance

- 🧠 **Bot Development**: Extend `BaseBot` to craft custom strategies
- 🔁 **Backtesting**: Plug-and-play CLI & code-based testing
- 🧬 **Optimization**: Use QPSO or LSGA to fine-tune parameters
- 📊 **Indicators**: Wrapped Tulip indicators for blazing performance
- 🌐 **Data Sources**: Pull candles from 100+ CEXs/DEXs with CCXT
- 📈 **Performance Metrics**: Evaluate bots with ROI, Sortino, Win Rate
- 🤖 **Speed**: Up to 50+ backtests/sec on low-end hardware

---

## ⚙️ Project Structure

```

qtradex/
├── core/             # Bot logic and backtesting
├── indicators/       # Technical indicators
├── optimizers/       # QPSO and LSGA
├── plot/             # Trade/metric visualization
├── private/          # Execution & paper wallets
├── public/           # Data feeds and utils
├── common/           # JSON RPC, BitShares nodes
└── setup.py          # Install script

```

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK.git QTradeX
cd QTradeX
pip install -e .
````

---

## 🧪 Example Bot: EMA Crossover

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
        fast = indicators["fast_ema"][-1]
        slow = indicators["slow_ema"][-1]
        if fast > slow:
            return Buy()
        elif fast < slow:
            return Sell()
        return Thresholds()

# Load data and run
data = qx.public.Data(exchange="kucoin", asset="BTC", currency="USDT", begin="2023-01-01", end="2023-12-31")
wallet = qx.private.PaperWallet({"BTC": 1, "USDT": 0})
bot = EMACrossBot()
qx.core.dispatch(bot, data, wallet)
```

🔗 See more bots in [QTradeX Algo Strategies](https://github.com/squidKid-deluxe/QTradeX-AI-Agents)

---

## 🚦 Usage Guide

| Step | What to Do                                                      |
| ---- | --------------------------------------------------------------- |
| 1️⃣  | Build a bot with custom logic by subclassing `BaseBot`          |
| 2️⃣  | Backtest using `qx.core.dispatch` + historical data             |
| 3️⃣  | Optimize with `qpso.py` or `lsga.py` (tunes stored in `/tunes`) |
| 4️⃣  | Deploy live                                                     |

---

## 🧭 Roadmap

* 🔄 **Live Execution** via CCXT & Bitshares
* 📈 More indicators (non-Tulip sources)
* 🏦 TradFi Connectors: Stocks, Forex, and Comex support

---

## 📚 Resources

* 🧠 [QTradeX Algo Trading Strategies](https://github.com/squidKid-deluxe/qtradex-ai-agents)
* 📘 [Tulipy Docs](https://tulipindicators.org)
* 🌍 [CCXT Docs](https://docs.ccxt.com)

---

## 📜 License

**WTFPL** — Do what you want. Just be awesome about it 😎

---

## ⭐ Star History

<a href="https://www.star-history.com/#squidKid-deluxe/QTradeX-Algo-Trading-SDK&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=squidKid-deluxe/QTradeX-Algo-Trading-SDK&type=Date" />
 </picture>
</a>

---

✨ Ready to start? Clone the repo, run your first bot, and let’s build the future of trading bots.
