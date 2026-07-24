"""
# HLOCV  DexPaprika On-Chain DEX Pools
# input (asset, currency, start, stop, period)
# output 'data' dictionary of numpy arrays
# 60, 300, 600, 900, 1800, 3600, 21600, 43200, 86400 second candle sizes
# data['unix']   # discretely spaced integers
# data['high']   # float
# data['low']    # float
# data['open']   # float
# data['close']  # float
# data['volume'] # float
# up to 366 candles per request; paginated for longer ranges
# on-chain DEX candle data across 36 chains, no API key required
# to learn more about available data visit these links
# https://docs.dexpaprika.com  https://api.dexpaprika.com/networks
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except, too-many-locals, too-many-arguments
#
# STANDARD MODULES
import time
from calendar import timegm

# THIRD PARTY MODULES
import numpy as np
import requests

# EXTINCTION EVENT MODULES
from qtradex.public.utilities import BadTimeframeError, clip_to_time_range

# ======================================================================
VERSION = "klines_dexpaprika v1.0.0"
API = "api.dexpaprika.com"  # free, no API key required
# ======================================================================
ATTEMPTS = 5
TIMEOUT = 30
MAX_LIMIT = 366  # candles per request, enforced by the API

# candle size in seconds -> DexPaprika interval string
INTERVALS = {
    60: "1m",
    300: "5m",
    600: "10m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    21600: "6h",
    43200: "12h",
    86400: "24h",
}


def parse_pool(pool):
    """
    A DexPaprika pool is identified by its network and pool address, since the
    same token pair can trade in many pools on many chains. Accept either a
    "network/address" (or "network:address") string, or a (network, address)
    tuple/list.

        pool="ethereum/0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
        pool=("solana", "3ne4mWqdYuNiYrYZC9TrA3FcfuFdErghH97vNPbjicr1")
    """
    if pool is None:
        raise ValueError(
            "DexPaprika requires a pool, e.g. "
            "pool='ethereum/0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640'. "
            "Find pool addresses at https://api.dexpaprika.com/networks/{network}/pools/search"
        )
    if isinstance(pool, (tuple, list)):
        network, address = pool[0], pool[1]
    else:
        sep = "/" if "/" in pool else ":"
        network, _, address = str(pool).partition(sep)
        if not address:
            raise ValueError(
                f"Could not parse pool {pool!r}; expected 'network/address'."
            )
    return network.strip(), address.strip()


def to_unix(iso):
    """DexPaprika timestamps look like '2026-07-23T00:00:00Z'; return epoch int."""
    return timegm(time.strptime(iso.replace("Z", "GMT"), "%Y-%m-%dT%H:%M:%S%Z"))


def fetch_page(network, address, start_unix, interval):
    """One OHLCV request; returns a list of candle dicts (may be empty)."""
    url = f"https://{API}/networks/{network}/pools/{address}/ohlcv"
    params = {"start": int(start_unix), "interval": interval, "limit": MAX_LIMIT}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": f"qtradex-{VERSION}"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 429:
                time.sleep(2 * attempt)
                continue
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and "message" in body:
                raise ValueError(f"DexPaprika: {body['message']}")
            return body
        except ValueError:
            raise
        except Exception:
            if attempt == ATTEMPTS:
                raise
            time.sleep(2 * attempt)
    return []


def klines_dexpaprika(asset, currency, start, end, interval, pool):
    """
    Input and output normalized requests for on-chain DEX candle data.
    Returns a dict with numpy array values for the keys
    ["high", "low", "open", "close", "volume", "unix"]
    where unix is int and the remainder are float, ideal for talib / tulip.

    `asset` and `currency` are used only for logging; the `pool` argument
    identifies the exact on-chain pool the candles come from.
    """
    if interval not in INTERVALS:
        raise BadTimeframeError(
            f"DexPaprika candle size {interval}s unsupported.",
            sorted(INTERVALS.keys()),
        )
    iv = INTERVALS[interval]
    network, address = parse_pool(pool)

    if end is None:
        end = int(time.time())
    if start is None:
        start = end - 10 * interval

    print(f"Collecting {asset}/{currency} candles from dexpaprika {network} {address} @ {iv}")

    rows = []
    seen = set()
    cursor = int(start)
    while cursor < end:
        page = fetch_page(network, address, cursor, iv)
        if not page:
            break
        fresh = 0
        for candle in page:
            unix = to_unix(candle["time_open"])
            if unix in seen:
                continue
            seen.add(unix)
            fresh += 1
            rows.append(
                {
                    "unix": unix,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                }
            )
        last_unix = to_unix(page[-1]["time_open"])
        # stop when the page is short (no more history) or we passed `end`
        if len(page) < MAX_LIMIT or last_unix >= end or fresh == 0:
            break
        cursor = last_unix + interval
        time.sleep(0.3)  # be polite to the shared free tier

    rows.sort(key=lambda r: r["unix"])
    data = {
        "unix": np.array([r["unix"] for r in rows], dtype=np.int64),
        "high": np.array([r["high"] for r in rows]),
        "low": np.array([r["low"] for r in rows]),
        "open": np.array([r["open"] for r in rows]),
        "close": np.array([r["close"] for r in rows]),
        "volume": np.array([r["volume"] for r in rows]),
    }
    return clip_to_time_range(data, int(start), int(end))
