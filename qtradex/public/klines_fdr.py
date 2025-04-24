from datetime import datetime

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from qtradex.public.stock_exchange_lookup import EXCHANGE_LOOKUP
from qtradex.public.utilities import BadTimeframeError


def klines_fdr(currency, asset, start, end, interval, _):
    if interval != 86400:
        raise BadTimeframeError("Invalid interval, daily candles only:", (86400,))

    exchange, asset = asset.split(":")

    start = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
    end = datetime.fromtimestamp(end).strftime("%Y-%m-%d")

    if (base_currency := EXCHANGE_LOOKUP.get(exchange, None)) is not None:
        # if the currency is the same as the one in the lookup, there is no need to get
        # the cross rate, so set it to one.
        cross_rate = 1
        if base_currency != currency:
            # but if they aren't, then we need to fetch the cross rate from fdr
            pass
    else:
        raise ValueError(
            f"{exchange} is not present in the lookup table.  "
            "Please check your spelling or open an issue on the QTradeX GitHub "
            "if you're sure this exchange exists."
        )

    try:
        df = fdr.DataReader(asset, start, end)
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

    # Print the DataFrame to inspect its structure
    print(df)

    # Check if the DataFrame is empty
    if df.empty:
        print("No data returned for the given parameters.")
        raise

    # Ensure the index is a DatetimeIndex
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
        print("Not a dataframe... fixing")

    # Prepare the data dictionary
    try:
        data = {
            "unix": np.array([int(idx.timestamp()) for idx in df.index]),
            "high": df["High"].to_numpy(),
            "low": df["Low"].to_numpy(),
            "open": df["Open"].to_numpy(),
            "close": df["Close"].to_numpy(),
            "volume": df.get(
                "Volume", np.zeros(len(df))
            ).to_numpy(),
        }
    except KeyError as e:
        print(
            f"KeyError: {e}. Check if the expected columns are present in the DataFrame."
        )
        raise

    print(data)
    return data
