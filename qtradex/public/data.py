import json
import math
import os
import time
from datetime import datetime

import ccxt
import numpy as np
from qtradex.common.json_ipc import json_ipc
from qtradex.common.utilities import it
from qtradex.core.quant import filter_glitches
from qtradex.public.klines_alphavantage import (klines_alphavantage_crypto,
                                                klines_alphavantage_forex,
                                                klines_alphavantage_stocks)
from qtradex.public.klines_bitshares import klines_bitshares
from qtradex.public.klines_ccxt import klines_ccxt
from qtradex.public.klines_cryptocompare import klines_cryptocompare
from qtradex.public.klines_synthetic import klines_synthetic
from qtradex.public.utilities import clip_to_time_range, implied, invert

DETAIL = True


def quantize_unix(unix_array, candle_size):
    # Quantize the unix times by the given candle size
    return np.floor(unix_array / candle_size) * candle_size


def merge_candles(candles1, candles2, candle_size):
    # Quantize the unix times for both dictionaries
    candles1["unix"] = quantize_unix(candles1["unix"], candle_size)
    candles2["unix"] = quantize_unix(candles2["unix"], candle_size)

    # Find the unique unix values from both dictionaries
    unique_unix = np.unique(np.concatenate([candles1["unix"], candles2["unix"]]))

    # Initialize the merged dictionary
    merged_candles = {
        "unix": unique_unix,
        "high": np.zeros_like(unique_unix, dtype=float),
        "low": np.zeros_like(unique_unix, dtype=float),
        "open": np.zeros_like(unique_unix, dtype=float),
        "close": np.zeros_like(unique_unix, dtype=float),
        "volume": np.zeros_like(unique_unix, dtype=float),
    }

    # Merge and handle conflicts by prioritizing candles1
    for i, unix in enumerate(unique_unix):
        # Initialize high, low, open, close for this unix
        high_vals = []
        low_vals = []
        vol_vals = []
        open_val = None
        close_val = None

        # Handle candles1 prices at the current unix time
        if unix in candles1["unix"]:
            idx = np.where(candles1["unix"] == unix)[0][0]
            high_vals.append(candles1["high"][idx])
            low_vals.append(candles1["low"][idx])
            vol_vals.append(candles1["volume"][idx])
            if open_val is None:
                open_val = candles1["open"][idx]
            close_val = candles1["close"][idx]  # Always overwrite with the last value

        # Handle candles2 prices at the current unix time
        if unix in candles2["unix"]:
            idx = np.where(candles2["unix"] == unix)[0][0]
            high_vals.append(candles2["high"][idx])
            low_vals.append(candles2["low"][idx])
            vol_vals.append(candles2["volume"][idx])
            if open_val is None:
                open_val = candles2["open"][idx]
            close_val = candles2["close"][idx]  # Always overwrite with the last value

        # Set the values for the merged dictionary
        merged_candles["high"][i] = np.max(high_vals) if high_vals else 0
        merged_candles["low"][i] = np.min(low_vals) if low_vals else 0
        merged_candles["open"][i] = open_val if open_val is not None else 0
        merged_candles["close"][i] = close_val if close_val is not None else 0
        merged_candles["volume"][i] = np.max(vol_vals) if vol_vals else 0

    return merged_candles


class Data:
    """
    Gather backtest data.
    """

    def __init__(
        self,
        exchange,
        asset,
        currency,
        begin,
        end=None,
        days=None,
        candle_size=86400,
        pool=None,
        api_key=None,
        intermediary=None,
    ):
        """
        See type(self) for accurate signature.
        """
        # Parse begin and end timestamps as given by user
        if days is not None and end is not None:
            raise ValueError("`days` OR `end` may be given, not both.")

        try:
            self.begin = time.mktime(datetime.strptime(begin, "%Y-%m-%d").timetuple())
        except Exception as error:
            raise ValueError("`begin` must be in %Y-%m-%d format.") from error

        if end is None and days is not None:
            self.end = int(self.begin - (86400 * days))
        elif days is None and end is not None:
            try:
                self.end = time.mktime(datetime.strptime(end, "%Y-%m-%d").timetuple())
            except Exception as error:
                raise ValueError("`end` must be in %Y-%m-%d format.") from error
        else:
            # Default to now
            self.end = int(time.time())

        # Add constants to self space
        self.exchange = exchange
        self.asset = asset
        self.currency = currency
        self.pool = pool
        self.candle_size = int(candle_size)
        self.begin = math.floor(self.begin / self.candle_size) * self.candle_size
        self.end = math.floor(self.end / self.candle_size) * self.candle_size
        self.api_key = api_key

        if self.pool is not None and exchange != "bitshares":
            raise ValueError(
                "Cannot get liquidity pool data for non-bitshares exchange."
            )

        self.raw_candles = {}

        self.intermediary = intermediary

        if intermediary is None:
            self.raw_candles = self.retrieve_and_cache_candles(
                self.asset, self.currency
            )
        else:
            print(f"Using {intermediary} to create implied price...")
            self.raw_candles = implied(
                self.retrieve_and_cache_candles(self.asset, self.intermediary),
                self.retrieve_and_cache_candles(self.intermediary, self.currency),
            )

        if not np.any(self.raw_candles["unix"]):
            raise RuntimeError(
                f"{self.exchange} does not provide {self.asset}/{self.currency} for this time range."
            )

        self.raw_candles["unix"] = quantize_unix(
            self.raw_candles["unix"], self.candle_size
        )

        self.begin = np.min(self.raw_candles["unix"])
        self.end = np.max(self.raw_candles["unix"])

    def __repr__(self):
        """
        <Data object>({candles} candles of data from {exchange}; {begin} to {end}; last price is {last})
        """
        return Data.__repr__.__doc__.strip("\n ").format(
            candles=len(self.raw_candles["close"]),
            exchange=self.exchange,
            begin=time.ctime(self.begin),
            end=time.ctime(self.end),
            last=self.raw_candles["close"][-1],
        )

    def __getitem__(self, index):
        return self.raw_candles[index]

    def keys(self):
        return self.raw_candles.keys()

    def values(self):
        return self.raw_candles.values()

    def items(self):
        return self.raw_candles.items()

    def retrieve_and_cache_candles(self, asset, currency):
        """
        Retrieves and caches candlestick data for the specified exchange, asset, and currency
        over a given time range. If the data for the requested time range already exists in cache,
        it will either merge the cached data with new data or use the cache entirely, depending on
        the overlap between the cached and requested time ranges. The method ensures that only the
        relevant data is gathered, merged, and returned, while also maintaining a persistent index
        for future reference.

        Steps:
        1. Check if the data index exists and load it. If not, initialize a new index.
        2. Construct a unique key based on the exchange, asset, and currency.
        3. If data for the given key is not in the cache, gather new data.
        4. If data is cached, determine the overlap with the requested time range and handle merging
           or retrieving the necessary data.
        5. Merge new and cached data if needed, clip the data to the requested time range,
           and update the cache.
        6. Update the data index to reflect the time range of the newly fetched data.

        Side Effects:
        - Writes new or updated data to `data_index.json` and `"{index_key} candles.json"`.
        - The `self.raw_candles` attribute is updated with the relevant data.

        Raises:
        - RuntimeError: If an unexpected condition occurs during the time range checks (should not happen).
        """
        # try to get the index, otherwise initialize it
        try:
            index = json_ipc("data_index.json")
        except FileNotFoundError:
            json_ipc("data_index.json", "{}")
            index = {}
        index_key = str((self.exchange, self.candle_size, asset, currency))
        rev_index_key = str((self.exchange, self.candle_size, currency, asset))
        total_time = [self.begin, self.end]
        raw_candles = None
        # if the exchange hasn't been queried before for this pair or its inverse
        if index_key not in index and rev_index_key not in index:
            # gather data
            raw_candles = self.gather_data(self.begin, self.end, asset, currency)
        else:
            inverted = rev_index_key in index
            index_key = rev_index_key if inverted else index_key
            # localize
            time_range = index[index_key]
            gather = None
            use_cache = None
            erase_cache = False
            # completely before or after what we need
            if time_range[1] < self.begin or time_range[0] > self.end:
                gather = [self.begin, self.end]
                # FIXME if we don't erase the cache here it would leave data gaps
                #       but that should be detected and filled, not overwritten
                erase_cache = True

            # within the range of what we need
            elif time_range[0] > self.begin and time_range[1] < self.end:
                # FIXME technically this is "helpful" because we could mix'n'match
                #       and get the data "around" what we already have, but that's
                #       a lot to implement.  We'll get there.
                gather = [self.begin, self.end]

            # covers the end of what we need but not the beginning
            elif time_range[0] > self.begin and time_range[1] >= self.end:
                gather = [self.begin, time_range[0] + self.candle_size]
                use_cache = [time_range[0], self.end]

            # covers the beginning of what we need but not the end
            elif time_range[0] <= self.begin and time_range[1] < self.end:
                gather = [time_range[1] - self.candle_size, self.end]
                use_cache = [self.begin, time_range[1]]

            # all of what we need and potentially more
            elif time_range[0] <= self.begin and time_range[1] >= self.end:
                use_cache = [self.begin, self.end]

            else:
                raise RuntimeError(
                    f"THIS SHOULD NOT HAPPEN!  Debug info: {time_range}, {self.begin} {self.end}"
                )

            data = []
            print(f"gather: {gather}  use_cache: {use_cache}")
            # gather up data from the two sources
            if gather is not None:
                data.append(self.gather_data(*gather, asset, currency))
            if use_cache is not None:
                data.append(
                    {
                        k: np.array(v)
                        for k, v in json_ipc(f"{index_key} candles.json").items()
                    }
                )
                if inverted:
                    data[-1] = invert(data[-1])

            candles = dict()
            if len(data) == 2:
                # merge the two data sources
                # this will implicitly never happen if erase_cache is True, though
                # an explicit check might be prudent
                candles = merge_candles(*data, self.candle_size)
            else:
                candles = data[0]

            raw_candles = candles
            if not erase_cache:
                total_time = [
                    min(time_range[0], self.begin),
                    max(time_range[1], self.end),
                ]
            else:
                total_time = [self.begin, self.end]

        # write the new cache with all the data
        json_ipc(
            f"{index_key} candles.json",
            json.dumps({k: v.tolist() for k, v in raw_candles.items()}),
        )

        # clip the data to the requested amount
        raw_candles = clip_to_time_range(raw_candles, self.begin, self.end)

        # stow the index
        if total_time is not None:
            index[index_key] = total_time
            json_ipc("data_index.json", json.dumps(index))

        return raw_candles

    def gather_data(self, begin, end, asset, currency, inverted=False):
        """
        Gathers historical candlestick data for a specified asset and currency pair
        from a variety of supported exchanges and APIs. The method checks the
        `self.exchange` attribute to determine the correct data source and fetches
        the data for the provided time range (`begin` to `end`). It supports a wide
        range of exchanges and data providers, including centralized exchanges,
        BitShares, CryptoCompare, Alpha Vantage, and others.

        Args:
            begin (int): The start of the time range for the candlestick data,
                         typically a Unix timestamp.
            end (int): The end of the time range for the candlestick data,
                       typically a Unix timestamp.

        Returns:
            dict: A dictionary containing the raw candlestick data, where the
                  keys may vary depending on the exchange (e.g., "unix", "open",
                  "high", "low", "close", "volume").

        Raises:
            ValueError: If the exchange is not in the list of supported exchanges.

        Notes:
            - The method supports a variety of exchanges including KuCoin, Kraken,
              Bittrex, Poloniex, Bitfinex, Binance, Coinbase, and others.
            - For some exchanges (e.g., `bitshares`, `cryptocompare`, `alphavantage_*`),
              specific API functions are used to fetch the data.
            - A `FIXME` comment indicates that the `nomics` API is no longer functional,
              and thus this part of the code may not work.
        """
        try:
            if DETAIL:
                print(f"gathering data from {self.exchange}, {begin} to {end}")
            if self.exchange.startswith("bitshares"):
                raw_candles = klines_bitshares(
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                    self.pool,
                )
            elif self.exchange == "cryptocompare":
                raw_candles = klines_cryptocompare(
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                    self.api_key,
                )
                # print(raw_candles)
                # days = len(self.raw_candles["unix"])
                # days = filter_glitches(days, tune)
            elif self.exchange == "alphavantage_stocks":
                raw_candles = klines_alphavantage_stocks(
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                    self.api_key,
                )
            elif self.exchange == "alphavantage_forex":
                raw_candles = klines_alphavantage_forex(
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                    self.api_key,
                )
            elif self.exchange == "alphavantage_crypto":
                raw_candles = klines_alphavantage_crypto(
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                    self.api_key,
                )
            elif self.exchange == "synthetic":
                raw_candles = klines_synthetic()
            elif self.exchange in ccxt.exchanges:
                print("Using CCXT...")
                raw_candles = klines_ccxt(
                    self.exchange,
                    asset,
                    currency,
                    begin,
                    end,
                    self.candle_size,
                )
            else:
                raise ValueError(f"Invalid exchange {self.exchange}")
            return raw_candles
        # FIXME go through the other klines scripts and have them raise BadSymbol
        #       instead of IndexErrors or KeyErrors
        # ccxt.base.errors.BadSymbol:
        except Exception:
            if inverted:
                raise
            print(
                it(
                    "yellow",
                    "Data gathering failed!  Reversing pair and trying again...",
                )
            )
            # reverse pair and try again
            asset, currency = currency, asset
            return invert(self.gather_data(begin, end, asset, currency, inverted=True))
