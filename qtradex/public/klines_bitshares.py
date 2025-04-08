"""
╔═╗─┐ ┬┌┬┐┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌╔═╗┬  ┬┌─┐┌┐┌┌┬┐
║╣ ┌┴┬┘ │ │││││   │ ││ ││││║╣ └┐┌┘├┤ │││ │ 
╚═╝┴ └─ ┴ ┴┘└┘└─┘ ┴ ┴└─┘┘└┘╚═╝ └┘ └─┘┘└┘ ┴ 

# HLOCV  BitShares DEX Orderbooks And Pools  
# input (asset, currency, start, stop, period)
# output 'data' dictionary of numpy arrays
# 86400, 14400, 300 second candle sizes
# data['unix'] # discretely spaced integers
# data['high'] # linearly interpolated float
# data['low'] # linearly interpolated float
# data['open'] # linearly interpolated float
# data['close'] # linearly interpolated float
# data['volume'] # float
# get up to 1000 candles
# provide statistical mean data from several nodes in network
# encapsulated websocket call in time out multiprocess
# interprocess communication via txt; returns numpy arrays
# normalized, discrete, interpolated data arrays
# to learn more about available data visit these links
# blocksights.info bitshares.network
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except, bad-continuation, too-many-arguments, too-many-locals
#
# STANDARD MODULES
import time
import traceback
from calendar import timegm
from collections import Counter
from datetime import datetime
from json import dumps as json_dumps
from json import loads as json_loads
from multiprocessing import Process, Value
from random import shuffle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
# THIRD PARTY MODULES
import numpy as np
# EXTINCTION EVENT MODULES
from qtradex.common.bitshares_nodes import bitshares_nodes
from qtradex.common.json_ipc import json_ipc
from qtradex.common.utilities import from_iso_date, to_iso_date, trace
from qtradex.public.kibana import fetch_candles
from qtradex.public.rpc import (rpc_get_objects, rpc_last,
                                rpc_lookup_asset_symbols, rpc_market_history,
                                wss_handshake)
from qtradex.public.utilities import BadTimeframeError

# ======================================================================
VERSION = "klines_bitshares v0.00000001"
API = "BitShares Public API Nodes"
# ======================================================================
ATTEMPTS = 30
TIMEOUT = 480
DETAIL = False

def parse_market_history(
    graphene_history: List[Dict], period: int, precision: Dict[str, int]
) -> List[Dict[str, float]]:
    """
    Convert from integer blockchain format to float data.

    This function processes market history data and converts various quantities (e.g., volumes and prices) from
    integer format to float, applying the specified precision for currency and asset values. It calculates the
    VWAP and returns the processed history in a structured format.

    Args:
        graphene_history (List[Dict]): The list of market history data (candles) to process.
        period (int): A time period adjustment to apply to the 'unix' timestamp.
        precision (Dict[str, int]): A dictionary containing precision for 'currency' and 'asset' volumes.

    Returns:
        List[Dict[str, float]]: A list of processed market history, each item containing the high, low, open, close,
                                 VWAP, currency volume, and asset volume.
    """
    cpr = precision["currency"]  # Precision for the quote currency
    apr = precision["asset"]  # Precision for the base asset

    history = []
    for candle in graphene_history:
        # Convert quote and base volumes to floats using the precision
        cvol = float(int(candle["quote_volume"])) / 10**cpr
        avol = float(int(candle["base_volume"])) / 10**apr

        # Calculate the VWAP (Volume Weighted Average Price)
        vwap = cvol / avol

        # Calculate and store the processed candle data
        history.append(
            {
                "high": (float(int(candle["high_quote"])) / 10**cpr)
                / (float(int(candle["high_base"])) / 10**apr),
                "low": (float(int(candle["low_quote"])) / 10**cpr)
                / (float(int(candle["low_base"])) / 10**apr),
                "open": (float(int(candle["open_quote"])) / 10**cpr)
                / (float(int(candle["open_base"])) / 10**apr),
                "close": (float(int(candle["close_quote"])) / 10**cpr)
                / (float(int(candle["close_base"])) / 10**apr),
                "unix": int(from_iso_date(candle["key"]["open"]))
                + period,  # Adjust Unix timestamp with period
                "vwap": vwap,
                "currency_v": cvol,
                "asset_v": avol,
            }
        )
    return history


def interpolate_previous(
    rpc, pair: str, data: Dict[str, List[float]], start: int, stop: int, period: int
) -> Dict[str, List[float]]:
    """
    Converts discrete data (buckets) into continuous klines (candlestick) format.

    This function interpolates market data to fill in missing intervals between `start` and `stop`
    with the specified `period`. If there are no data points for certain time intervals, the function
    fills those with the previous available data (or a default value if no prior data exists).

    Args:
        rpc: The RPC client used to fetch the last known close price if no data is available.
        pair (str): The market pair for which data is being processed (e.g., 'BTC/USD').
        data (Dict[str, List[float]]): A dictionary containing the market data with 'unix' timestamps
                                       and corresponding 'volume', 'high', 'low', 'open', and 'close' values.
        start (int): The start timestamp (in Unix format) for the interval.
        stop (int): The end timestamp (in Unix format) for the interval.
        period (int): The time period (in seconds) for each data bucket.

    Returns:
        Dict[str, List[float]]: A dictionary containing the processed market data with interpolated or filled
                                 values for 'high', 'low', 'open', 'close', 'volume', and 'unix' timestamps.
    """
    # Ensure start, stop, and period are integers
    start = int(start)
    stop = int(stop)
    period = int(period)

    # Initialize a dictionary to store the new data with interpolated values
    data2 = {
        "volume": [],
        "high": [],
        "low": [],
        "open": [],
        "close": [],
        "unix": [
            *range(max(int(min(data["unix"])), start), stop + period, period)
        ],  # Generate timestamps for each period
    }

    # If there is existing data to interpolate from
    if data["unix"]:
        for unix in data2["unix"]:
            match = False  # Flag to check if data was found for the current timestamp

            # Loop through the provided data to find a match for the current timestamp
            for idx, _ in enumerate(data["unix"]):
                diff = unix - data["unix"][idx]
                if 0 <= diff < period:
                    # If the data point fits within this period, add it to the new data
                    data2["volume"].append(data["volume"][idx])
                    data2["high"].append(data["high"][idx])
                    data2["low"].append(data["low"][idx])
                    data2["open"].append(data["open"][idx])
                    data2["close"].append(data["close"][idx])
                    match = True
                    break  # Exit the inner loop since we found a match

            # If no match was found for the timestamp, fill with previous data or default value
            if not match:
                if unix == start:
                    # If it's the first candle, set the close price as the first data point's close
                    close = data["close"][0]
                else:
                    # Otherwise, interpolate the previous close value
                    close = data2["close"][-1]

                # Fill in the missing data points with the interpolated value
                data2["volume"].append(0)
                data2["high"].append(close)
                data2["low"].append(close)
                data2["open"].append(close)
                data2["close"].append(close)

    else:
        # If no data exists, fetch the last known close price from RPC
        close = rpc_last(rpc, pair)
        if DETAIL:
            print(">" * 30, close, "<" * 30)

        # Fill all time periods with the last known close price
        for unix in data2["unix"]:
            data2["volume"].append(0)
            data2["high"].append(close)
            data2["low"].append(close)
            data2["open"].append(close)
            data2["close"].append(close)

    # Return the newly created dictionary with interpolated or filled data
    return {
        "high": data2["high"],
        "low": data2["low"],
        "open": data2["open"],
        "close": data2["close"],
        "volume": data2["volume"],
        "unix": data2["unix"],
    }


def reformat(data: List[Dict[str, float]]) -> Dict[str, List[float]]:
    """
    Switch from a list of dictionaries to a dictionary of lists.

    This function takes a list of dictionaries, where each dictionary represents a data point
    with fields like 'unix', 'high', 'low', 'open', 'close', and 'currency_v'. It transforms this
    list into a dictionary where each key corresponds to a list of values for that field.

    Args:
        data (List[Dict[str, float]]): A list of dictionaries containing market data. Each dictionary
                                       includes keys like 'unix', 'high', 'low', 'open', 'close', and 'currency_v'.

    Returns:
        Dict[str, List[float]]: A dictionary where each key corresponds to a list of values from the input data.
                                 The keys include 'unix', 'high', 'low', 'open', 'close', and 'volume'.
    """
    # Initialize a dictionary to hold the transformed data
    data2: Dict[str, List[float]] = {
        "unix": [],
        "high": [],
        "low": [],
        "open": [],
        "close": [],
        "volume": [],
    }

    # Iterate through each item in the input data and populate the corresponding lists in data2
    for entry in data:
        data2["unix"].append(entry["unix"])
        data2["high"].append(entry["high"])
        data2["low"].append(entry["low"])
        data2["open"].append(entry["open"])
        data2["close"].append(entry["close"])
        data2["volume"].append(
            entry["currency_v"]
        )  # Assuming 'currency_v' is the volume

    return data2


def normalize(data: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Eliminate extreme outlier data by normalizing the high, low, open, and close prices.

    This function processes market data and normalizes the values for high, low, open, and close
    prices by removing extreme outliers. The values are adjusted based on their relationships with
    the open and close prices, ensuring that no values are excessively far from the average.

    Args:
        data (Dict[str, List[float]]): A dictionary containing market data with keys for 'high', 'low',
                                       'open', and 'close', each mapping to a list of float values.

    Returns:
        Dict[str, List[float]]: The normalized market data, with high, low, open, and close values adjusted
                                 to eliminate extreme outliers.
    """
    # Iterate over each index in the close price list (assuming all lists have the same length)
    for idx in range(len(data["close"])):
        # Calculate the high and low based on the max and min of open, close, high, and low values
        high = max(
            data["high"][idx], data["low"][idx], data["open"][idx], data["close"][idx]
        )
        low = min(
            data["high"][idx], data["low"][idx], data["open"][idx], data["close"][idx]
        )

        # Normalize the high and low values by capping them at the calculated limits
        data["high"][idx] = high
        data["low"][idx] = low

        # Filter out extreme candles, ensuring they are within 0.5x to 2x the open/close average
        open_close_avg = data["open"][idx] + data["close"][idx]

        data["high"][idx] = min(data["high"][idx], open_close_avg)
        data["open"][idx] = min(data["open"][idx], open_close_avg)
        data["close"][idx] = min(data["close"][idx], open_close_avg)

        # Adjust the low, open, and close values to be no lower than 1/4th of the open/close average
        low_bound = open_close_avg / 4

        data["low"][idx] = max(data["low"][idx], low_bound)
        data["open"][idx] = max(data["open"][idx], low_bound)
        data["close"][idx] = max(data["close"][idx], low_bound)

    return data


def truncate_depth(data: Dict[str, List[float]], depth: int) -> Dict[str, List[float]]:
    """
    Truncate the data to the specified depth.

    This function reduces each list of market data (e.g., 'unix', 'high', 'low', etc.) to the most recent
    `depth` number of elements, effectively truncating the data to the requested depth.

    Args:
        data (Dict[str, List[float]]): A dictionary containing market data, with keys like 'unix',
                                       'high', 'low', 'open', 'close', and 'volume'.
        depth (int): The number of elements to retain from the end of each list.

    Returns:
        Dict[str, List[float]]: The truncated market data with each list reduced to the specified depth.
    """
    for key in data:
        data[key] = data[key][-depth:]
    return data


def dexdata(
    signal,
    asset: str,
    currency: str,
    _,
    start: int,
    stop: int,
    period: int,
    depth: int,
) -> None:
    """
    Fetch and process market data from multiple nodes, then return the common kline response.

    This function contacts several blockchain nodes to gather market data for the specified
    asset and currency pair over the given time period. It fetches data from multiple sources,
    normalizes it, and then returns the common data after filtering out outliers and interpolating.

    Args:
        signal (multiprocessing.Value): A shared value that signals when the function is complete.
        asset (str): The asset symbol (e.g., "BTC").
        currency (str): The currency symbol (e.g., "USD").
        start (int): The start Unix timestamp for the data retrieval period.
        stop (int): The stop Unix timestamp for the data retrieval period.
        period (int): The period in seconds for each data point (e.g., 300 for 5-minute candles).
        depth (int): The depth to truncate the data to.

    Raises:
        ValueError: If the provided period is not valid.
    """
    if period not in [300, 14400, 86400]:
        json_ipc("proxy.txt", '["failed"]')
        signal.value = 2
        return

    window = int(period * 200)  # window size for each node call
    calls = min(
        5, (1 + int((stop - start) / float(window)))
    )  # how many times to call nodes
    if DETAIL:
        print("window", window, "calls", calls)

    # Fetch and shuffle node list to distribute load
    nodes = bitshares_nodes
    shuffle(nodes)

    mavens = []  # Holds potential datasets from multiple nodes
    now = int(time.time()) + 1

    while True:
        try:
            # Rotate the nodes list for load balancing
            nodes.append(nodes.pop(0))
            node = nodes[0]
            if DETAIL:
                print("node", node)

            rpc = wss_handshake(node)  # Establish a websocket connection

            # Fetch asset and currency information
            ret = rpc_lookup_asset_symbols(rpc, asset, currency)
            asset_id = ret[0]["id"]
            currency_id = ret[1]["id"]
            precision = {"currency": ret[1]["precision"], "asset": ret[0]["precision"]}

            data = []
            # Fetch market data in chunks from multiple calls
            for i in range((calls - 1), -1, -1):
                g_start = now - (i + 1) * window
                g_stop = now - i * window
                if DETAIL:
                    print("call", i, period, window, g_start, g_stop)
                data += rpc_market_history(
                    rpc, currency_id, asset_id, period, g_start, g_stop
                )

            # Convert raw blockchain data to human-readable candles
            data = parse_market_history(data, period, precision)
            # Ensure data is sorted by Unix timestamp
            data = sorted(data, key=lambda k: k["unix"])

            # Track duplicate data to ensure consensus
            mavens.append(json_dumps(data))
            max_count = Counter(mavens).most_common(1)[0][1]
            if DETAIL:
                print(max_count, node)

            rpc.close()  # Close the RPC connection

            # Break the loop if 3 identical datasets are found
            if max_count == 3:
                break

        except Exception as error:
            trace(error)
            continue

    # Transform the data from a list of dicts to a dict of lists
    data = reformat(data)

    # Normalize and filter extreme values
    data = normalize(data)

    # Interpolate missing price data in buckets with no action
    data = interpolate_previous(rpc, f"{asset}:{currency}", data, start, stop, period)

    # Limit the data to the requested depth
    data = truncate_depth(data, depth)

    # Save the processed data to a file
    json_ipc("proxy.txt", json_dumps(data))

    # Signal completion of the process
    signal.value = 1


def dexdata_pools(
    signal,
    asset: str,
    currency: str,
    pool: str,
    start: int,
    stop: int,
    period: int,
    depth: int,
) -> None:
    """
    Fetch market data for a specific liquidity pool and process it into kline format.

    This function contacts several blockchain nodes to gather market data for a liquidity pool, processes
    the data (normalizes, interpolates missing data, etc.), and returns the common kline response.

    Args:
        signal (multiprocessing.Value): A shared value used for signaling completion of the function.
        asset (str): The asset symbol (e.g., "BTC").
        currency (str): The currency symbol (e.g., "USD").
        pool (str): The liquidity pool identifier (e.g., "BTC_USD_POOL").
        start (int): The start Unix timestamp for the data retrieval period.
        stop (int): The stop Unix timestamp for the data retrieval period.
        period (int): The period in seconds for each data point (e.g., 300 for 5-minute candles).
        depth (int): The depth to truncate the data to.

    Raises:
        ValueError: If the asset and currency pair in the pool are invalid or mismatched.
    """
    # Get node connection
    if DETAIL:
        print("Getting node...")
    rpc = wss_handshake()

    # Verify the asset and currency pair in the pool
    if DETAIL:
        print("Checking pair...")
    getobj = rpc_get_objects(rpc, pool)
    asset_a = rpc_get_objects(rpc, getobj["asset_a"])["symbol"]
    asset_b = rpc_get_objects(rpc, getobj["asset_b"])["symbol"]

    # Ensure that the asset and currency match one of the pool's assets
    assert asset_a in [asset, currency] and asset_b in [
        asset,
        currency,
    ], f"Invalid pair, must be {asset_a}:{asset_b} or its inversion, got {asset}:{currency}"

    # Determine if the pair needs inversion
    inverted = asset_a == asset

    # Fetch market data from the pool
    data = fetch_candles(rpc, pool, start, stop, period)

    # Process and normalize the data
    data = reformat(data)  # Convert from list of dicts to dict of lists
    data = normalize(data)  # Filter extreme outlier values

    # Interpolate missing data
    data = interpolate_previous(rpc, f"{asset}:{currency}", data, start, stop, period)

    # Truncate data to the requested depth
    data = truncate_depth(data, depth)

    # Invert the data if necessary
    if inverted:
        data["low"], data["high"] = data["high"], data["low"]

    # Invert prices if required (excluding 'unix' and 'volume')
    data = {
        k: ([1 / i for i in v] if inverted and k not in ["unix", "volume"] else v)
        for k, v in data.items()
    }

    # Write processed data to a file
    json_ipc("proxy.txt", json_dumps(data))

    # Signal completion
    signal.value = 1


def klines_bitshares(
    asset: str, currency: str, start: int, stop: int, period: int, pool: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    Retrieves kline data for the specified asset and currency from BitShares.

    This function calculates the kline (candlestick) data by contacting relevant nodes,
    performing necessary preprocessing, and returning the results as a dictionary of numpy arrays.

    Args:
        asset (str): The asset symbol (e.g., "BTC").
        currency (str): The currency symbol (e.g., "USD").
        start (int): The start Unix timestamp for the data retrieval period.
        stop (int): The stop Unix timestamp for the data retrieval period.
        period (int): The period in seconds for each data point (e.g., 300 for 5-minute candles).
        pool (Optional[str]): The liquidity pool identifier (optional).

    Returns:
        dict: A dictionary containing the kline data with keys: 'unix', 'high', 'low', 'open', 'close', 'volume'.
              Each key's value is a numpy array of the respective data.

    Raises:
        ValueError: If the depth is less than 1 or if there's an issue with the data retrieval.
    """
    begin = time.time()

    # Adjust the start and stop times
    start = int(start)
    period = int(period)
    stop = min(int(stop), int(time.time()))

    # Calculate depth based on the requested period and time window
    depth = int((stop - start) / float(period))
    if not pool and depth > 1000:
        start = int(stop - (period * 1000))
        depth = 1000

    # Calculate the duration in days
    days = (stop - start) / 86400.0
    if DETAIL:
        print(
            f"RPC {asset} {currency} {start}s {stop}e CANDLE {period}s DAYS {days:.1f} DEPTH {depth}"
        )

    # Ensure depth is valid
    if depth < 1:
        raise ValueError("Depth is less than 1, invalid request.")

    # Fetch data using multiprocessing with retries
    data = ""
    while not data:
        signal = Value("i", 0)
        attempt = 0
        while attempt < ATTEMPTS and not signal.value:
            attempt += 1
            if DETAIL:
                print(f"\nklines_bitshares attempt: {attempt} at {time.ctime()}")

            # Choose which function to call based on whether a pool is specified
            child = Process(
                target=dexdata_pools if pool else dexdata,
                args=(signal, asset, currency, pool, start, stop, period, depth),
            )
            child.daemon = False
            child.start()
            child.join(TIMEOUT)
            child.terminate()

        # Retrieve data written by child process or return stale data in case of failure
        data = json_ipc("proxy.txt")

    # Clear the proxy file after reading the data
    json_ipc("proxy.txt", "")

    if "failed" in data:
        raise BadTimeframeError("Invalid period. Must be one of", [300, 14400, 86400])

    # Convert data to numpy arrays for performance
    if DETAIL:
        print(f"klines_bitshares elapsed: {time.time() - begin:.2f} seconds")
    return {k: np.array(v) for k, v in data.items()}
