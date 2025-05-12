"""
# Daily HLOCV US Stock, Forex, Crypto:Forex from alphavantage.com API
" litepresence 2019 "
" ********** ALPHA RELEASE TO PUBLIC DOMAIN WITH NO WARRANTY ********* "
# input (asset, currency, start, stop, period)
# output 'data' dictionary of numpy arrays
# daily candles only, 86400 spacing normalized
# data['unix'] # integers
# data['high'] # float
# data['low'] # float
# data['open'] # float
# data['close'] # float
# data['volume'] # float
# up to 20 years daily historical data
# stock weekends removed for discrete spacing
# encapsulated http call in time out multiprocess
# interprocess communication via txt; returns numpy arrays
# to learn more about available data visit these links
crypto_list = "https://www.alphavantage.co/digital_currency_list"
fiat_list = "https://www.alphavantage.co/physical_currency_list"
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except, too-many-arguments
#
# STANDARD MODULES
import time
from json import dumps as json_dumps
from multiprocessing import Process, Value

# THIRD PARTY MODULES
import numpy as np
import requests

# EXTINCTION EVENT MODULES
from qtradex.common.json_ipc import json_ipc

# GLOBAL CONSTANTS
VERSION = "alphavantage v0.00000001"
API = "www.alphavantage.com"  # GET FREE API KEY HERE
FULL = True
ATTEMPTS = 3
TIMEOUT = 60


def from_iso_date(date):
    """
    returns unix epoch given YYYY-MM-DD
    """
    return int(time.mktime(time.strptime(str(date), "%Y-%m-%d")))


def to_list_of_dicts_stocks(ret):
    """
    change from dict of dicts to list of dicts;
    rename keys, add unix
    """
    data = []
    for key, val in ret.items():
        val["unix"] = from_iso_date(key)
        val["open"] = val.pop("1. open")
        val["high"] = val.pop("2. high")
        val["low"] = val.pop("3. low")
        val["close"] = val.pop("4. close")
        val["volume"] = val.pop("6. volume")
        val["split"] = val.pop("8. split coefficient")
        data.append(val)
    # sort list of dicts by value of key unix
    return sorted(data, key=lambda k: k["unix"])


def to_list_of_dicts_forex(ret):
    """
    change from dict of dicts to list of dicts;
    rename keys, add unix
    """
    data = []
    for key, val in ret.items():
        val["unix"] = from_iso_date(key)
        val["open"] = val.pop("1. open")
        val["high"] = val.pop("2. high")
        val["low"] = val.pop("3. low")
        val["close"] = val.pop("4. close")
        val["volume"] = 1
        data.append(val)
    # sort list of dicts by value of key unix
    return sorted(data, key=lambda k: k["unix"])


def to_list_of_dicts_crypto(ret, currency):
    """
    change from dict of dicts to list of dicts;
    rename keys, add unix
    """
    data = []
    for key, val in ret.items():
        val["unix"] = from_iso_date(key)
        val["open"] = val.pop(f"1a. open ({currency})")
        val["high"] = val.pop(f"2a. high ({currency})")
        val["low"] = val.pop(f"3a. low ({currency})")
        val["close"] = val.pop(f"4a. close ({currency})")
        val["volume"] = val.pop("5. volume")
        data.append(val)
    # sort list of dicts by value of key unix
    return sorted(data, key=lambda k: k["unix"])


def to_dict_of_lists(data):
    """
    switch from list of dicts <-> to dict of lists
    """
    data2 = {"unix": [], "high": [], "low": [], "open": [], "close": [], "volume": []}
    try_split = True
    for idx, _ in enumerate(data):  # days):
        if try_split:
            try:
                split = float(data[idx]["split"])
            except Exception:
                split = 1
                try_split = False
        else:
            split = 1
        data2["unix"].append(int(data[idx]["unix"]))
        data2["high"].append(float(data[idx]["high"]) * split)
        data2["low"].append(float(data[idx]["low"]) * split)
        data2["open"].append(float(data[idx]["open"]) * split)
        data2["close"].append(float(data[idx]["close"]) * split)
        data2["volume"].append(float(data[idx]["volume"]) * split)
    return data2


def remove_weekends(data):
    """
    remove weekends for backtesting discrete time
    """
    depth = len(data["unix"])
    end = max(data["unix"])
    begin = end - depth * 86400
    data["unix"] = [*range(begin, end, 86400)]
    return data


def window(start, stop, data):
    """
    limit data to window requested
    """
    data2 = {"unix": [], "high": [], "low": [], "open": [], "close": [], "volume": []}
    for i in range(len(data["unix"])):
        if start <= data["unix"][i] <= stop:
            data2["unix"].append(int(data["unix"][i]))
            data2["high"].append(float(data["high"][i]))
            data2["low"].append(float(data["low"][i]))
            data2["open"].append(float(data["open"][i]))
            data2["close"].append(float(data["close"][i]))
            data2["volume"].append(float(data["volume"][i]))
    return data2


def stocks(signal, asset, currency, start, stop, period, api_key):
    """
    make api call for stock data
    """
    days = int((stop - start) / float(period))
    if api_key is None:
        raise ValueError("YOU MUST GET API KEY FROM alphavantage.com")
    if period != 86400:
        raise ValueError("Daily Candles Only")
    outputsize = "compact"
    if FULL:
        outputsize = "full"
    uri = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "outputsize": outputsize,
        "symbol": asset,
        "apikey": api_key,
    }
    ret = requests.get(uri, params=params, timeout=(6, 30)).json()
    ret = ret["Time Series (Daily)"]
    # post process to extinctionEVENT format
    data = to_list_of_dicts_stocks(ret)
    data = to_dict_of_lists(data)
    data = remove_weekends(data)
    data = window(start, stop, data)
    # inter process communication via txt
    json_ipc("proxy.txt", json_dumps(data))
    signal.value = 1


def forex(signal, asset, currency, start, stop, period, api_key):
    """
    make api call for forex data
    """
    days = int((stop - start) / float(period))
    if api_key is None:
        raise ValueError("YOU MUST GET API KEY FROM alphavantage.com")
    if period != 86400:
        raise ValueError("Daily Candles Only")
    outputsize = "compact"
    if FULL:
        outputsize = "full"
    uri = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_DAILY",
        "outputsize": outputsize,
        "from_symbol": currency,
        "to_symbol": asset,
        "apikey": api_key,
    }
    ret = requests.get(uri, params=params, timeout=(6, 30)).json()
    ret = ret["Time Series FX (Daily)"]
    # post process to extinctionEVENT format
    data = to_list_of_dicts_forex(ret)
    data = to_dict_of_lists(data)
    data = remove_weekends(data)
    data = window(start, stop, data)
    # inter process communication via txt
    json_ipc("proxy.txt", json_dumps(data))
    signal.value = 1


def crypto(signal, asset, currency, start, stop, period, api_key):
    """
    make api call for crypto data
    """
    days = int((stop - start) / float(period))
    if api_key is None:
        raise ValueError("YOU MUST GET API KEY FROM alphavantage.com")
    if period != 86400:
        raise ValueError("Daily Candles Only")
    outputsize = "compact"
    if FULL:
        outputsize = "full"
    uri = "https://www.alphavantage.co/query"
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "outputsize": outputsize,
        "market": currency,
        "symbol": asset,
        "apikey": api_key,
    }
    ret = requests.get(uri, params=params, timeout=(6, 30)).json()
    print(ret)
    ret = ret["Time Series (Digital Currency Daily)"]
    # post process to extinctionEVENT format
    data = to_list_of_dicts_crypto(ret, currency)
    data = to_dict_of_lists(data)
    data = remove_weekends(data)
    data = window(start, stop, data)
    # inter process communication via txt
    json_ipc("proxy.txt", json_dumps(data))
    signal.value = 1


def klines_alphavantage_stocks(
    asset, currency, start, stop, candle=86400, api_key=None
):
    """
    importable definition for klines
    """
    begin = time.time()
    start = int(start)
    stop = int(stop)
    period = int(candle)
    depth = int((stop - start) / 86400.0)
    print(
        "API AlphaVantage Stocks %s %s %ss %se CANDLE %s DAYS %s"
        % (asset, currency, start, stop, period, depth)
    )
    signal = Value("i", 0)
    i = 0
    while (i < ATTEMPTS) and not signal.value:
        i += 1
        print("")
        print("alphavantage api attempt:", i, time.ctime())
        child = Process(
            target=stocks, args=(signal, asset, currency, start, stop, period, api_key)
        )
        child.daemon = False
        child.start()
        child.join(TIMEOUT)
        child.terminate()
    # read text file written by child; in worst case return stale
    data = json_ipc("proxy.txt")
    json_ipc("proxy.txt", "")
    # convert dict values from lists to numpy arrays
    print("klines_alphavantage elapsed:", f"{time.time() - begin:.2f}")
    return {k: np.array(v) for k, v in data.items()}


def klines_alphavantage_forex(asset, currency, start, stop, candle=86400, api_key=None):
    """
    importable definition for klines
    """
    begin = time.time()
    start = int(start)
    stop = int(stop)
    period = int(candle)
    depth = int((stop - start) / 86400.0)
    print(
        "API AlphaVantage Forex %s %s %ss %se CANDLE %s DAYS %s"
        % (asset, currency, start, stop, period, depth)
    )
    signal = Value("i", 0)
    i = 0
    while (i < ATTEMPTS) and not signal.value:
        i += 1
        print("")
        print("alphavantage api attempt:", i, time.ctime())
        child = Process(
            target=forex, args=(signal, asset, currency, start, stop, period, api_key)
        )
        child.daemon = False
        child.start()
        child.join(TIMEOUT)
        child.terminate()
    # read text file written by child; in worst case return stale
    data = json_ipc("proxy.txt")
    json_ipc("proxy.txt", "")
    # convert dict values from lists to numpy arrays
    print("klines_alphavantage elapsed:", f"{time.time() - begin:.2f}")
    return {k: np.array(v) for k, v in data.items()}


def klines_alphavantage_crypto(
    asset, currency, start, stop, candle=86400, api_key=None
):
    """
    importable definition for klines
    """
    begin = time.time()
    start = int(start)
    stop = int(stop)
    period = int(candle)
    depth = int((stop - start) / 86400.0)
    print(
        "API AlphaVantage Crypto %s %s %ss %se CANDLE %s DAYS %s"
        % (asset, currency, start, stop, period, depth)
    )
    signal = Value("i", 0)
    i = 0
    while (i < ATTEMPTS) and not signal.value:
        i += 1
        print("")
        print("alphavantage api attempt:", i, time.ctime())
        child = Process(
            target=crypto, args=(signal, asset, currency, start, stop, period, api_key)
        )
        child.daemon = False
        child.start()
        child.join(TIMEOUT)
        child.terminate()
    # read text file written by child; in worst case return stale
    data = json_ipc("proxy.txt")
    json_ipc("proxy.txt", "")
    # convert dict values from lists to numpy arrays
    print("klines_alphavantage elapsed:", f"{time.time() - begin:.2f}")
    return {k: np.array(v) for k, v in data.items()}
