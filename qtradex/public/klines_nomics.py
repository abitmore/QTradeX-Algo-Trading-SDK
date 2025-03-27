"""
Daily HLOCV Crypto Exchange Specific Candles from nomics.com
litepresence 2019
********** ALPHA RELEASE TO PUBLIC DOMAIN WITH NO WARRANTY *********
# input (exchange, asset, currency, start, stop, period) *note exchange
# output 'data' dictionary of numpy arrays
# daily candles only, 86400
# data['unix'] # integers
# data['high'] # float
# data['low'] # float
# data['open'] # float
# data['close'] # float
# data['volume'] # float
# get candles to data origin
# exact data as reported by each exchange
# encapsulated websocket call in time out multiprocess
# interprocess communication via txt; returns numpy arrays
# to learn more about available data visit these links
market_list = "https://api.nomics.com/v1/markets?key=YOUR-API-KEY"
exchange_list = "https://nomics.com/exchanges"
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except, too-many-arguments
#
# STANDARD MODULES
import time
from calendar import timegm
from datetime import datetime
from json import dumps as json_dumps
from multiprocessing import Process, Value

# THIRD PARTY MODULES
import numpy as np
import requests

# EXTINCTION EVENT MODULES
from qtradex.common.json_ipc import json_ipc

# GLOBALS
VERSION = "klines_nomics v0.00000001"
API = "www.nomics.com"  # GET FREE API KEY HERE
ATTEMPTS = 3
TIMEOUT = 60


def to_iso_date(unix):
    """
    returns iso8601 datetime given unix epoch
    """
    return datetime.utcfromtimestamp(int(unix)).isoformat()


def from_iso_date(date):
    """
    returns unix epoch given iso8601 datetime
    """
    return int(timegm(time.strptime(str(date), "%Y-%m-%dT%H:%M:%SZ")))


def to_list_of_dicts(ret):
    """
    change from dict of dicts to list of dicts;
    rename keys, add unix
    """
    data = []
    for idx, _ in enumerate(ret):
        data2 = {}
        data2["unix"] = from_iso_date(ret[idx]["timestamp"])
        data2["open"] = ret[idx]["open"]
        data2["high"] = ret[idx]["high"]
        data2["low"] = ret[idx]["low"]
        data2["close"] = ret[idx]["close"]
        data2["volume"] = ret[idx]["volume"]
        data.append(data2)
    # sort list of dicts by value of key unix
    return sorted(data, key=lambda k: k["unix"])


def to_dict_of_lists(data):
    """
    switch from list of dicts <-> to dict of lists
    """
    data2 = {}
    data2["unix"] = []
    data2["high"] = []
    data2["low"] = []
    data2["open"] = []
    data2["close"] = []
    data2["volume"] = []
    try_split = True
    for idx, _ in enumerate(data):
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


def window(start, stop, data):
    """
    limit data to window requested
    """
    data2 = {}
    data2["unix"] = []
    data2["high"] = []
    data2["low"] = []
    data2["open"] = []
    data2["close"] = []
    data2["volume"] = []
    for idx in enumerate(data["unix"]):
        if start <= data["unix"][idx] <= stop:
            data2["unix"].append(int(data["unix"][idx]))
            data2["high"].append(float(data["high"][idx]))
            data2["low"].append(float(data["low"][idx]))
            data2["open"].append(float(data["open"][idx]))
            data2["close"].append(float(data["close"][idx]))
            data2["volume"].append(float(data["volume"][idx]))
    return data2


def market_syntax(exchange, asset, currency, api_key):
    url = f"https://api.nomics.com/v1/markets?key={api_key}&exchange={exchange}"
    ret = requests.get(url).json()
    print(ret)
    based = [i for i in ret if i["base"] == asset]
    quoted = [i for i in based if i["quote"] == currency]

    return quoted[0]["market"]


def crypto(signal, exchange, asset, currency, start, stop, period, api_key):
    """
    #
    """
    if api_key == "":
        raise ValueError("YOU MUST GET API KEY FROM nomics.com")
    if period != 86400:
        raise ValueError("Daily Candles Only")
    market = market_syntax(exchange, asset, currency, api_key)
    time.sleep(1.5)
    days = int((stop - start) / float(period))
    # make api call for data
    uri = "https://api.nomics.com/v1/exchange_candles"
    params = {
        "key": api_key,
        "interval": "1d",
        "exchange": exchange,
        "market": market,
        "start": to_iso_date(start),
        "end": to_iso_date(stop),
    }
    ret = requests.get(uri, params=params, timeout=(6, 30))
    print(ret)
    print(ret.text)
    ret = ret.json()
    print(ret)
    # post process to extinctionEVENT format
    data = to_list_of_dicts(ret)
    data = to_dict_of_lists(data)
    data = window(start, stop, data)
    # inter process communication via txt
    json_ipc("proxy.txt", json_dumps(data))
    signal.value = 1


def klines_nomics(exchange, asset, currency, start, stop, candle=86400, api_key=None):
    """
    importable definition
    """
    begin = time.time()
    start = int(start)
    stop = int(stop)
    period = int(candle)
    depth = int((stop - start) / 86400.0)
    print(
        "API Nomics %s %s %s %ss %se CANDLE %s DAYS %s"
        % (exchange, asset, currency, start, stop, period, depth)
    )
    signal = Value("i", 0)
    iteration = 0
    while (iteration < ATTEMPTS) and not signal.value:
        iteration += 1
        print("")
        print("klines_nomics attempt:", iteration, time.ctime())
        child = Process(
            target=crypto,
            args=(signal, exchange, asset, currency, start, stop, period, api_key),
        )
        child.daemon = False
        child.start()
        child.join(TIMEOUT)
        child.terminate()
    # read text file written by child; in worst case return stale
    data = json_ipc("proxy.txt")
    json_ipc("proxy.txt", "")
    # convert dict values from lists to numpy arrays
    print("klines_nomics elapsed:", f"{time.time() - begin:.2f}")
    return {k: np.array(v) for k, v in data.items()}
