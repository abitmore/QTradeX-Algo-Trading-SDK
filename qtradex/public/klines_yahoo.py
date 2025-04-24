import ccxt
import numpy as np
import yfinance as yf
from qtradex.common.utilities import format_timeframe


def klines_yahoo(currency, asset, start, end, interval, _):
    if asset == "USD":
        raise ccxt.BadSymbol("Asset cannot be USD")

    print(f"Downloading {asset} from {start} to {end}")
    df = yf.download(
        asset,
        int(start),
        int(end),
        format_timeframe(interval).replace("M", "mo").replace("w", "wk"),
    )

    data = {
        "unix": np.array([int(idx.timestamp()) for idx in df.index]),
        "high": df["High"].to_numpy().T[0],
        "low": df["Low"].to_numpy().T[0],
        "open": df["Open"].to_numpy().T[0],
        "close": df["Close"].to_numpy().T[0],
        "volume": df["Volume"].to_numpy().T[0],
    }
    return data
