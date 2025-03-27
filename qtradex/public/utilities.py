import numpy as np


def clip_to_time_range(candles, start_unix, end_unix):
    # Find the indices where 'unix' values fall within the specified range
    valid_indices = np.where(
        (candles["unix"] >= start_unix) & (candles["unix"] <= end_unix)
    )[0]
    # print(start_unix, end_unix, candles["unix"][0])
    # print(valid_indices)

    # Clip the dictionary by keeping only the valid indices
    clipped_candles = {
        "unix": candles["unix"][valid_indices],
        "high": candles["high"][valid_indices],
        "low": candles["low"][valid_indices],
        "open": candles["open"][valid_indices],
        "close": candles["close"][valid_indices],
        "volume": candles["volume"][valid_indices],
    }

    return clipped_candles


def invert(candles):
    """
    invert a chunk of kline data
    i.e. BTC/USDT becomes USDT/BTC
    """
    if isinstance(candles, list):
        return [
            {
                "unix": i["unix"],
                # high and low are flipped
                "high": 1 / i["low"],
                "low": 1 / i["high"],
                # but open and low remain the same
                "open": 1 / i["open"],
                "close": 1 / i["close"],
                "volume": i["volume"] * ((i["open"] + i["close"]) / 2),
            }
            for i in candles
        ]
    elif isinstance(candles, dict):
        return {
            "unix": np.array(candles["unix"]),
            # high and low are flipped
            "high": 1 / np.array(candles["low"]),
            "low": 1 / np.array(candles["high"]),
            # but open and low remain the same
            "open": 1 / np.array(candles["open"]),
            "close": 1 / np.array(candles["close"]),
            "volume": np.array(candles["volume"])
            * ((np.array(candles["open"]) + np.array(candles["close"])) / 2),
        }


def implied(candles1, candles2):
    """
    Take two sets of candles, in format
    {
        "unix": np.ndarray(np.float64),
        "high": np.ndarray(np.float64),
        "low": np.ndarray(np.float64),
        "open": np.ndarray(np.float64),
        "close": np.ndarray(np.float64),
        "volume": np.ndarray(np.float64),
    }
    and return the implied price.
    For example, if candles1 represents XRP/BTC
    and candles2 represents BTC/XLM,
    this function should return the implied candles for XRP/XLM.
    """

    # Ensure the datasets are the same length by truncating to the smaller length
    minlen = min(len(candles1["unix"]), len(candles2["unix"]))
    candles1 = {k: v[-minlen:] for k, v in candles1.items()}
    candles2 = {k: v[-minlen:] for k, v in candles2.items()}

    # Create a new dataset with synthesized data
    implied_candles = {
        "unix": [],
        "close": [],
        "high": [],
        "low": [],
        "open": [],
        "volume": [],
    }

    for idx in range(minlen):
        # Extract the necessary values for synthesis
        d1_h, d2_h = candles1["high"][idx], candles2["high"][idx]
        d1_l, d2_l = candles1["low"][idx], candles2["low"][idx]
        d1_o, d2_o = candles1["open"][idx], candles2["open"][idx]
        d1_c, d2_c = candles1["close"][idx], candles2["close"][idx]
        unix = candles1["unix"][idx]

        # Calculate synthesized close and open values
        _close = d1_c / d2_c
        _open = d1_o / d2_o

        # Use the synthesis strategy for high/low calculations
        _high, _low = synthesize_high_low(
            d1_h, d2_h, d1_l, d2_l, d1_o, d2_o, d1_c, d2_c
        )

        # Ensure high is the maximum and low is the minimum
        _low = min(_high, _low, _open, _close)
        _high = max(_high, _low, _open, _close)

        # Assuming 'volume' is the same in both datasets for the implied price.
        volume = candles1["volume"][idx]  # Assuming both have the same volume

        implied_candles["unix"].append(unix)
        implied_candles["close"].append(_close)
        implied_candles["high"].append(_high)
        implied_candles["low"].append(_low)
        implied_candles["open"].append(_open)
        implied_candles["volume"].append(volume)

    # Return the resulting dataset
    return {k: np.array(v) for k, v in implied_candles.items()}


def synthesize_high_low(d1_h, d2_h, d1_l, d2_l, d1_o, d2_o, d1_c, d2_c):
    """
    This function calculates the high and low values.

    Args:
        d1_h, d2_h, d1_l, d2_l, d1_o, d2_o, d1_c, d2_c (float): The values from both asset datasets.

    Returns:
        tuple: The calculated high and low values.
    """
    _high = (d1_h / d2_c) / 2 + (d1_c / d2_l) / 2
    _low = (d1_l / d2_c) / 2 + (d1_c / d2_h) / 2
    return _high, _low
