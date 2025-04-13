"""
Klines specific utilities
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from qtradex.common.utilities import rotate
from scipy.interpolate import CubicSpline

DETAIL = False


class BadTimeframeError(ValueError):
    pass


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


def quantize_unix(unix_array, candle_size):
    # Quantize the unix times by the given candle size
    return np.floor(unix_array / candle_size) * candle_size


def merge_candles(candles, candle_size):
    # Quantize the unix times for both dictionaries
    candles = [
        {**batch, "unix": quantize_unix(batch["unix"], candle_size)}
        for batch in candles
    ]

    # Find the unique unix values from both dictionaries
    unique_unix = np.sort(
        np.unique(np.concatenate([batch["unix"] for batch in candles]))
    )

    # Initialize the merged dictionary
    merged_candles = {
        "unix": unique_unix,
        "high": np.zeros_like(unique_unix, dtype=float),
        "low": np.zeros_like(unique_unix, dtype=float),
        "open": np.zeros_like(unique_unix, dtype=float),
        "close": np.zeros_like(unique_unix, dtype=float),
        "volume": np.zeros_like(unique_unix, dtype=float),
    }
    if any("candle_size" in i for i in candles):
        merged_candles["candle_size"] = np.zeros_like(unique_unix, dtype=float)

    # Merge and handle conflicts by prioritizing candles1
    for i, unix in enumerate(unique_unix):
        # Initialize high, low, open, close for this unix
        high_vals = []
        low_vals = []
        vol_vals = []
        open_val = None
        close_val = None
        candle_sizes = []

        for batch in candles:
            # Handle batch prices at the current unix time
            if unix in batch["unix"]:
                idx = np.where(batch["unix"] == unix)[0][0]
                high_vals.append(batch["high"][idx])
                low_vals.append(batch["low"][idx])
                vol_vals.append(batch["volume"][idx])
                if open_val is None:
                    open_val = batch["open"][idx]
                close_val = batch["close"][idx]
                if "candle_size" in batch:
                    candle_sizes.append(batch["candle_size"][idx])

        # Set the values for the merged dictionary
        if close_val is not None:
            merged_candles["high"][i] = np.max(high_vals)
            merged_candles["low"][i] = np.min(low_vals)
            merged_candles["open"][i] = open_val
            merged_candles["close"][i] = close_val
            merged_candles["volume"][i] = np.max(vol_vals)
        else:
            close_val = merged_candles["close"][-1]
            merged_candles["high"][i] = close_val
            merged_candles["low"][i] = close_val
            merged_candles["open"][i] = close_val
            merged_candles["close"][i] = close_val
            merged_candles["volume"][i] = 0
        if candle_sizes:
            merged_candles["candle_size"][i] = max(candle_sizes)

    return merged_candles


def interpolate(data, oldperiod, newperiod):
    """
    Interpolate a given dictionary of OHLCV candle data to a smaller candle size
    """
    # can't interpolate 0 or 1 datapoints
    if len(data["unix"]) <= 1:
        return data
    # create an array of the new timestamps
    newstamps = np.arange(data["unix"][0], data["unix"][-1], newperiod)

    return {
        "unix": newstamps,
        # make cubic interpolations for high, low, open, and close
        "high": CubicSpline(data["unix"], data["high"])(newstamps),
        "low": CubicSpline(data["unix"], data["low"])(newstamps),
        "open": CubicSpline(data["unix"], data["open"])(newstamps),
        "close": CubicSpline(data["unix"], data["close"])(newstamps),
        # volume needs to be a nearest (stepwise) interpolation;
        # scipy now recommends not using interp1d and references
        # their source code for nearest interpolation, adapted here.
        # note that the volume is divided by the difference in candle size so that
        # it will still sum to the same amount:
        "volume": (data["volume"] * newperiod / oldperiod)[
            # find the nearest indices to those requested
            np.searchsorted(data["unix"], newstamps, side="left")
            # clip them into range
            .clip(0, data["unix"].shape[0] - 1)
            # and make them int pointers to indices
            .astype(np.intp)
        ],
    }


def create_candles(data, width=86400, stride=600):
    candles = []
    data = np.array(data)

    # Initialize variables
    start_index = 0
    num_data_points = len(data)

    for unix in np.arange(data[0][0], data[-1][0], stride):
        # Move the start_index to the first valid data point for the current candle
        while start_index < num_data_points and data[start_index][0] < unix - width:
            start_index += 1

        # If there are no valid data points, continue to the next candle
        if start_index >= num_data_points or data[start_index][0] > unix:
            continue

        # Now we can slice the data for the current candle
        end_index = start_index
        while end_index < num_data_points and data[end_index][0] <= unix:
            end_index += 1

        # Get the valid data points for this candle
        datapoints = data[start_index:end_index]

        # Create the candle
        current_candle = {
            "open": datapoints[0][1],
            "high": np.max(datapoints[:, 1]),
            "low": np.min(datapoints[:, 1]),
            "close": datapoints[-1][1],
            "volume": np.sum(datapoints[:, 2]),
            "unix": unix,
        }
        if datapoints.shape[1] > 3:
            current_candle["candle_size"] = np.max(datapoints[:, 3])
        candles.append(current_candle)

    return {k: np.array(v) for k, v in rotate(candles).items()}


def reaggregate(data, candle_size, stride=None):
    discrete = list(
        itertools.chain(
            *[
                [
                    [unix, _open, volume / 4, size],
                    [unix, high, volume / 4, size],
                    [unix, low, volume / 4, size],
                    [unix, close, volume / 4, size],
                ]
                for unix, high, low, _open, close, volume, size in zip(
                    data["unix"],
                    data["high"],
                    data["low"],
                    data["open"],
                    data["close"],
                    data["volume"],
                    data.get(
                        "candle_size",
                        np.array(
                            [data["unix"][1] - data["unix"][0]] * len(data["unix"])
                        ),
                    ),
                )
            ]
        )
    )
    return create_candles(
        discrete, candle_size, stride if stride is not None else candle_size
    )


def fetch_composite_data(data, new_size):
    """
    Fetches and aggregates high-resolution candle data for a given asset from a
    specified exchange.

    This function retrieves high-resolution candle data based on the provided `data`
    object and the desired `new_size` for the candle size. If the high-resolution data
    does not cover the entire requested time range, the function will iteratively fetch
    lower-resolution data using progressively larger candle sizes until sufficient data
    is gathered. The fetched data is then interpolated to match the desired candle size
    and merged with the previously obtained data.

    Parameters:
    ----------
    data : Data
        An instance of a Data class containing the necessary parameters for fetching
        candle data, including exchange, asset, currency, begin and end timestamps,
        pool, API key, and intermediary.

    new_size : int
        The desired size of the candles (in seconds) for the high-resolution data.

    Returns:
    -------
    high_res : Data
        A Data class containing the aggregated high-resolution candle data, with the
        specified `new_size` for the candles.

    Notes:
    -----
    - This function handles potential circular imports by defining a local `Data` class
      based on the input `data`.
    - This function uses a predefined list of valid candle sizes (in seconds) to fetch
      additional data if needed.
    - The final aggregated data is assigned back to the original `data` object for
      compatibility with the rest of the QTradeX system.
    """
    end = data.end

    # ok, yes this is kind of hacky, but importing Data from here would cause a
    # circular import so let's just define a "Data class" from this one:
    Data = type(data)

    if DETAIL:
        print(f"Gathering data with candle_size={new_size}")
    try:
        high_res = Data(
            exchange=data.exchange,
            asset=data.asset,
            currency=data.currency,
            begin=data.begin,
            pool=data.pool,
            api_key=data.api_key,
            intermediary=data.intermediary,
            end=end,
            candle_size=new_size,
        ).raw_candles
    except KeyboardInterrupt:
        raise
    except:
        high_res = {}
    else:
        high_res = {
            **high_res,
            "candle_size": np.array([new_size] * len(high_res["unix"])),
        }
    # now we have high resolution data between high_res.begin and high_res.end
    # if that isn't all that we were asked for then we need to gather more data,
    # but this time with a bigger candle size so we can get more data.
    # However, the exchange may not have much more, thus, we need to check a steadily
    # increasing candle size until we get enough data, interpolating and appending to
    # high_res each time

    end = high_res["unix"][0] if high_res else end
    # minute, ten minute, hour, two hour, four hour, day, week
    valid_sizes = [
        60,
        10 * 60,
        60 * 60,
        60 * 60 * 2,
        60 * 60 * 4,
        86400,
        86400 * 7,
    ]
    idx = 0
    while (end > data.begin) and (idx <= len(valid_sizes) - 1):
        fetch_size = valid_sizes[idx]
        # iterate through all valid sizes but the hope is to break early
        if fetch_size <= new_size:
            idx += 1
            continue

        if DETAIL:
            print(
                f"Did not fetch enough data, trying again with interpolated {fetch_size} candles..."
            )
        # get data for this candle size, attempting to get all that we need
        try:
            fetched_data = Data(
                exchange=data.exchange,
                asset=data.asset,
                currency=data.currency,
                begin=data.begin,
                pool=data.pool,
                api_key=data.api_key,
                intermediary=data.intermediary,
                end=end,
                candle_size=fetch_size,
            ).raw_candles
        except KeyboardInterrupt:
            raise
        except:
            fetched_data = {}

        if fetched_data:
            # interpolate to match the candle size we actually wanted
            inter_data = interpolate(fetched_data, fetch_size, new_size)

            if high_res:
                # merge with the other data we already have
                high_res = merge_candles(
                    [
                        high_res,
                        {
                            **inter_data,
                            "candle_size": np.array(
                                [fetch_size] * len(inter_data["unix"])
                            ),
                        },
                    ],
                    new_size,
                )
            else:
                high_res = inter_data

            # find out new earliest point and set that to the end for the next query
            end = np.min(high_res["unix"])

        # and try the next biggest candle size
        idx += 1

    # now we need to make a set of candles with a window the same as what we were given
    # but with a step size the same as the requested candle size.
    # (This is similar to the FFT gathering process of OpenAI's Whisper model.)
    # This ensures that strategies that were built on one candle size (i.e. 86400) don't
    # malfunction when run on ten-minute candles during a live session.

    # to do this, we need to treat these candles as discrete data with four (ohlc) data
    # points at each unix.  Thus:

    # plt.ioff()
    # plt.scatter(high_res["unix"], high_res["open"], color="gray")
    # plt.scatter(high_res["unix"], high_res["high"], color="gray")
    # plt.scatter(high_res["unix"], high_res["low"], color="gray")
    # plt.scatter(high_res["unix"], high_res["close"], color="gray")

    re_agg = reaggregate(high_res, data.candle_size, stride=new_size)

    # plt.scatter(re_agg["unix"], re_agg["open"], color="red", marker="+")
    # plt.scatter(re_agg["unix"], re_agg["high"], color="orange", marker="+")
    # plt.scatter(re_agg["unix"], re_agg["low"], color="yellow", marker="+")
    # plt.scatter(re_agg["unix"], re_agg["close"], color="green", marker="+")
    # plt.show()

    # assign our aggregated data to the data class we were given so that it's
    # in the format the rest of QTradeX expects it to be in
    data.raw_candles = re_agg
    data.candle_size = new_size
    return data, high_res
