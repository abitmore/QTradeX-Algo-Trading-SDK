"""
Daily HLOCV 3000+ Altcoin:Altcoin Candle Data from cryptocompare.com
litepresence 2019 "
********** ALPHA RELEASE TO PUBLIC DOMAIN WITH NO WARRANTY *********
input (asset, currency, start, stop, period)
output 'data' dictionary of numpy arrays
daily candles only, 86400
data['unix'] # integers
data['high'] # float
data['low'] # float
data['open'] # float
data['close'] # float
data['volume'] # float
up to 2000 candles of daily historical data
synthesizes altcoin1/altcoin2 and altcoin/fiat data
encapsulated http call in time out multiprocess
interprocess communication via txt; returns numpy arrays
normalize data arrays: high never more than 2x; low no less than 0.5x
to learn more about available data visit this link
https://www.cryptocompare.com/coins/list/USD/1
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except, too-many-locals, too-many-arguments
#
# STANDARD MODULES
import time
from json import dumps as json_dumps
from multiprocessing import Process, Value
from typing import Dict, List

# THIRD PARTY MODULES
import numpy as np
import requests

# EXTINCTION EVENT MODULES
from qtradex.common.json_ipc import json_ipc

# ======================================================================
VERSION = "klines_cryptocompare v0.00000001"
API = "www.cryptocompare.com"  # GET FREE API KEY HERE
# ======================================================================
SYNTHESIS = 4
ATTEMPTS = 3
TIMEOUT = 60


def synthesize(
    signal, asset: str, currency: str, start: int, stop: int, period: int, api_key
) -> None:
    """
    Synthesizes a dataset for altcoin/altcoin pairs by converting them into a synthetic asset pair.
    The synthesis uses BTC as a base to convert the price data.
    The SYNTHESIS parameter determines the type of calculation for the high/low price.

    Args:
        signal (multiprocessing.Value): The signal used for interprocess communication.
        asset (str): The asset symbol (e.g., 'ETH').
        currency (str): The currency symbol (e.g., 'BTC').
        start (int): The start timestamp (unix format).
        stop (int): The stop timestamp (unix format).
        period (int): The period of the candles in seconds (e.g., 86400 for daily candles).

    Returns:
        None: The function directly modifies the `signal` value and passes the reformatted data.
    """

    # Fetch the data for the different assets
    if currency == "BTC" or (currency in ["USD", "CNY"] and asset == "BTC"):
        data = cryptocompare(asset, currency, start, stop, period, api_key)
    else:
        dataset1 = cryptocompare("BTC", asset, start, stop, period, api_key)
        dataset2 = cryptocompare("BTC", currency, start, stop, period, api_key)

        # Ensure the datasets are the same length by truncating to the smaller length
        minlen = min(len(dataset1), len(dataset2))
        dataset1 = dataset1[-minlen:]
        dataset2 = dataset2[-minlen:]

        # Create a new dataset with synthesized data
        dataset3 = []

        for idx in range(minlen):
            d1x, d2x = dataset1[idx], dataset2[idx]

            # Extract the necessary values for synthesis
            d1_h, d2_h = d1x["high"], d2x["high"]
            d1_l, d2_l = d1x["low"], d2x["low"]
            d1_o, d2_o = d1x["open"], d2x["open"]
            d1_c, d2_c = d1x["close"], d2x["close"]
            unix = d1x["unix"]

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

            # Capture the volume information
            volumefrom = d1x["volumefrom"]
            volumeto = d1x["volumeto"]

            # Create the new candle
            candle = {
                "unix": unix,
                "close": _close,
                "high": _high,
                "low": _low,
                "open": _open,
                "volumefrom": volumefrom,
                "volumeto": volumeto,
            }

            # Append the synthesized candle to the new dataset
            dataset3.append(candle)

        # Use the reformatted data and pass it to the reformat function
        data = dataset3

    data = clip_beginning_of_time(data)
    reformat(signal, data)


def clip_beginning_of_time(data):
    for candle in data[:]:
        if candle["volumefrom"]:
            break
        data.pop(0)
    return data


def synthesize_high_low(
    d1_h: float,
    d2_h: float,
    d1_l: float,
    d2_l: float,
    d1_o: float,
    d2_o: float,
    d1_c: float,
    d2_c: float,
) -> (float, float):
    """
    This function calculates the high and low values based on different synthesis strategies.

    Args:
        d1_h, d2_h, d1_l, d2_l, d1_o, d2_o, d1_c, d2_c (float): The values from both asset datasets.

    Returns:
        tuple: The calculated high and low values.
    """
    # Select synthesis strategy
    if SYNTHESIS == 0:
        _high = d1_h / d2_c
        _low = d1_l / d2_c
    elif SYNTHESIS == 1:
        _high = d1_h / d2_l
        _low = d1_l / d2_h
    elif SYNTHESIS == 2:
        _high = d1_h / d2_h
        _low = d1_l / d2_l
    elif SYNTHESIS == 3:
        _high = ((d1_h + d1_c) / 2) / ((d2_l + d2_c) / 2)
        _low = ((d1_l + d1_c) / 2) / ((d2_h + d2_c) / 2)
    elif SYNTHESIS == 4:
        _high = (d1_h / d2_c) / 2 + (d1_c / d2_l) / 2
        _low = (d1_l / d2_c) / 2 + (d1_c / d2_h) / 2
    else:
        raise ValueError("Invalid SYNTHESIS value")

    return _high, _low


def cryptocompare(
    asset: str, currency: str, start: int, stop: int, period: int, api_key
) -> List[Dict[str, any]]:
    """
    Fetch historical daily candles from CryptoCompare API.

    Args:
        asset (str): The asset symbol (e.g., 'BTC').
        currency (str): The currency symbol (e.g., 'USD').
        start (int): The start timestamp (unix format).
        stop (int): The end timestamp (unix format).
        period (int): The candle period in seconds (must be 86400 for daily candles).

    Returns:
        List[Dict[str, any]]: A list of candles, each containing time, close, high, low, open, volumefrom, and volumeto.

    Raises:
        ValueError: If API key is missing or an invalid period is provided.
    """

    # Ensure the API key is available
    if api_key is None:
        raise ValueError("You must get an API key from cryptocompare.com.")

    # Only support daily candles (86400 seconds period)
    if period != 86400:
        raise ValueError("Only daily candles (86400 seconds) are supported.")

    # API endpoint
    uri = "https://min-api.cryptocompare.com/data/histoday"

    # Prepare parameters
    fsym = asset
    tsym = currency
    candles = int((stop - start) / float(period))
    limit = min(10 + candles, 2000)
    calls = (candles // 2000) + 1  # Calculate the number of API calls required

    # Initialize data list
    data = []

    # Current timestamp in seconds
    now = int(time.time())

    # Make multiple API calls for required depth
    for i in range(calls - 1, -1, -1):
        # Calculate the timestamp for each call
        tots = now - i * 2000 * 86400
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "limit": limit,
            "aggregate": 1,
            "toTs": tots,
        }

        headers = {"Apikey": api_key}

        try:
            # Make the request to CryptoCompare
            response = requests.get(
                uri, params=params, headers=headers, timeout=(6, 30)
            )
            response.raise_for_status()  # Raise an HTTPError if the response code is 4xx/5xx
            ret = response.json()

            # Filter and add valid candles to the data list
            data += [candle for candle in ret["Data"] if candle["close"] > 0]

        except requests.exceptions.RequestException as error:
            print(f"Request failed: {error}. Retrying...")
            continue

    # Slice the data to match the requested range
    return [
        {(k if k != "time" else "unix"): v for k, v in i.items()}
        for i in data[-candles:]
    ]


def reformat(signal, data: List[Dict[str, any]]) -> None:
    """
    Converts a list of candles (dicts) into a dictionary of lists for each attribute.

    Args:
        signal (Value): The multiprocessing signal used to indicate completion.
        data (list): A list of dictionaries where each dictionary contains candle data.

    Returns:
        None: The function modifies the 'data' dictionary in place and normalizes it after reformatting.
    """
    # Initialize the dictionary of lists
    data2 = {
        "unix": [candle["unix"] for candle in data],
        "high": [candle["high"] for candle in data],
        "low": [candle["low"] for candle in data],
        "open": [candle["open"] for candle in data],
        "close": [candle["close"] for candle in data],
        "volume": [candle["volumefrom"] for candle in data],
    }

    # Normalize the reformatted data
    normalize(signal, data2)


def normalize(signal, data: Dict[str, list]) -> None:
    """
    Normalize high, low, open, and close prices in the dataset.

    This function adjusts the high, low, open, and close values to ensure they stay within reasonable limits
    based on their relationships and ensures extreme price fluctuations are corrected.

    Args:
        signal (Value): The multiprocessing signal used to indicate completion.
        data (dict): A dictionary containing 'high', 'low', 'open', 'close', and other data as lists.

    Returns:
        None: The function directly modifies the `data` dictionary in place.
    """
    for idx in range(len(data["close"])):
        # Get the maximum and minimum of high, low, open, and close values
        high = max(
            data["high"][idx], data["low"][idx], data["open"][idx], data["close"][idx]
        )
        low = min(
            data["high"][idx], data["low"][idx], data["open"][idx], data["close"][idx]
        )

        # Set high and low to the calculated values
        data["high"][idx] = high
        data["low"][idx] = low

        # Apply filtering for extreme price fluctuations based on open/close average
        open_close_avg = (data["open"][idx] + data["close"][idx]) / 2

        # Apply the 0.5X to 2X range for high/low values and clamp extreme values accordingly
        data["high"][idx] = min(data["high"][idx], open_close_avg * 2)
        data["low"][idx] = max(data["low"][idx], open_close_avg / 2)

        # Ensure open/close values are within the same range
        data["open"][idx] = min(data["open"][idx], open_close_avg * 2)
        data["close"][idx] = min(data["close"][idx], open_close_avg * 2)

        # Ensure low prices are not too low by restricting them to 0.25x the open/close average
        data["low"][idx] = max(data["low"][idx], open_close_avg / 4)
        data["open"][idx] = max(data["open"][idx], open_close_avg / 4)
        data["close"][idx] = max(data["close"][idx], open_close_avg / 4)

    # Save the normalized data to a file
    json_ipc("proxy.txt", json_dumps(data))

    # Indicate that the process is complete
    signal.value = 1


def trace(error):
    """
    traceback message
    """
    msg = str(type(error).__name__) + str(error.args)
    return msg


def klines_cryptocompare(
    asset: str, currency: str, start: int, stop: int, period: int, api_key
) -> Dict[str, np.ndarray]:
    """
    Retrieves kline data from CryptoCompare API.

    This function fetches the kline (candlestick) data for the specified asset and currency,
    performs necessary preprocessing, and returns the results as a dictionary of numpy arrays.

    Args:
        asset (str): The asset symbol (e.g., "BTC").
        currency (str): The currency symbol (e.g., "USD").
        start (int): The start Unix timestamp for the data retrieval period.
        stop (int): The stop Unix timestamp for the data retrieval period.
        period (int): The period in seconds for each data point (e.g., 300 for 5-minute candles).

    Returns:
        dict: A dictionary containing the kline data with keys: 'unix', 'high', 'low', 'open', 'close', 'volume'.
              Each key's value is a numpy array of the respective data.

    Raises:
        ValueError: If no data is retrieved or if there's an issue with the process.
    """
    begin = time.time()

    # Ensure start, stop, and period are valid integers
    start, stop, period = map(int, [start, stop, period])

    # Calculate the number of days in the requested period
    days = (stop - start) / 86400.0
    print(
        f"API cryptocompare {asset} {currency} {start}s {stop}e CANDLE {period}s DAYS {days:.1f}"
    )

    # Signal for interprocess communication
    signal = Value("i", 0)
    attempt = 0

    # Attempt to fetch data multiple times if necessary
    while attempt < ATTEMPTS and not signal.value:
        attempt += 1
        print(f"\nklines_cryptocompare attempt: {attempt} at {time.ctime()}")

        # Start child process to retrieve data
        child = Process(
            target=synthesize,
            args=(signal, asset, currency, start, stop, period, api_key),
        )
        child.daemon = False
        child.start()
        child.join(TIMEOUT)
        child.terminate()

    # Retrieve data from the file written by the child process
    data = json_ipc("proxy.txt")
    json_ipc("proxy.txt", "")  # Clear the file after reading

    # Ensure data is returned in the correct format as numpy arrays
    if not data:
        raise ValueError("No data retrieved, check logs for errors.")

    print(f"klines_cryptocompare elapsed: {time.time() - begin:.2f} seconds")
    # exit()

    return {k: np.array(v) for k, v in data.items()}
