# pylint: disable=broad-except, unspecified-encoding, too-many-locals
"""
╔═╗─┐ ┬┌┬┐┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌╔═╗┬  ┬┌─┐┌┐┌┌┬┐
╠╣ ┌┴┬┘ │ │││││   │ ││ ││││╠╣ └┐┌┘├┤ │││ │ 
╚═╝┴ └─ ┴ ┴┘└┘└─┘ ┴ ┴└─┘┘└┘╚═╝ └┘ └─┘┘└┘ ┴ 

make requests to kibana to get historical data
and parse the discrete data that is received into candles
"""
import datetime

# STANDARD MOUDLES
import json
import time
from bisect import bisect

import aiohttp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import requests
from qtradex.common.bitshares_nodes import bitshares_nodes
from qtradex.common.utilities import it, to_iso_date
from qtradex.public.kibana_queries import kibana_swaps
from qtradex.public.rpc import precision, wss_handshake


def make_candles(data, interval):
    """
    make candles from discrete data and perform
    operations on them
    """
    interval = int(interval * 1000)
    # data = [[unix, price, volume], ...]
    data = discrete_to_candles(data, interval)
    # [{}, {}, {}]
    data = {
        "unix": [i["unix"] for i in data],
        "high": [i["high"] for i in data],
        "low": [i["low"] for i in data],
        "open": [i["open"] for i in data],
        "close": [i["close"] for i in data],
        "volume": [i["volume"] for i in data],
    }
    return [
        {
            "unix": int(unix / 1000),
            "open": data["open"][idx],
            "high": data["high"][idx],
            "low": data["low"][idx],
            "close": data["close"][idx],
            "currency_v": data["volume"][idx],
        }
        for idx, unix in enumerate(data["unix"])
    ]


def clip_beginning_of_time(data):
    for candle in data[:]:
        if candle["currency_v"]:
            break
        data.pop(0)
    print(data)
    exit()
    return data


def get_trade_history(rpc, market, start, stop):
    """
    Retrieve the trade history from kibana
    for a specified market within the specified time range.
    Parameters:
    market (str): The market name.
    start (int): Start time in Unix timestamp format.
    stop (int): Stop time in Unix timestamp format.
    Returns:
    list: A list of trade history data, if any was found, else an empty list
    """
    start = start
    stop = stop
    final = []
    pdata = {}

    while True:
        params = kibana_swaps(market, start, stop)
        # Send the kibana query
        response = requests.get(
            "https://es.bts.mobi/bitshares-*/_search",
            data=json.dumps(params),
            headers={"Content-Type": "application/json"},
        )
        data = response.json()
        # print(data)
        # Stop if there's no new data,
        # the data is the same as the previous query,
        # or if the batch size has been reached
        if (
            # if there are no hits
            (len(data["hits"]["hits"]) <= 1)
            # or we got the same data back
            or (pdata == data)
        ):
            # job done.
            break
        pdata = data
        data = parse_price_history(rpc, data)

        if len(data) == 1:
            break

        if (not data) and (int(start) != int(stop)):
            print(
                it(
                    "yellow",
                    "[WARNING]: KIBANA: Length of return list is 0 with start and stop"
                    f" {to_iso_date(start)} and {to_iso_date(stop)}",
                )
            )
        else:
            print(
                it(
                    "green",
                    f"[DEBUG]: KIBANA: {len(data)} data"
                    f" point{'s' if len(data) > 1 else ''} collected",
                )
            )
        final.extend(data)
        stop = sorted(data, key=lambda x: x[0])[0][0] / 1000
        # Wait before sending another query
        time.sleep(1)
    return final


def format_raw_history_data(data):
    """
    make the returned kibana data indexing for swaps and fills identical
    """
    data = json.loads(
        json.dumps(data)
        .replace("pays", "paid")
        .replace("receives", "received")
        .replace("operation_history.operation_result.keyword", "operation_history.op")
    )
    data = {
        i["sort"][0]: {
            **{
                k: v[0] if isinstance(v, list) else v
                for k, v in json.loads(
                    i["fields"]["operation_history.op"][0].replace("\\", "")
                )[1].items()
                if k in ["paid", "received"]
            },
            "account": i["fields"]["account_history.account.keyword"][0],
            "blocknum": i["fields"]["block_data.block_num"][0],
            "op_id": i["fields"]["account_history.operation_id"][0],
        }
        for i in data["hits"]["hits"]
    }
    return data


def process_operations_history(rpc, data):
    """
    Process raw operations history data and convert the amounts to a readable format.
    Args:
        data (dict): Raw data of operations history.
    Returns:
        dict: Operations history data with amounts converted to a readable format.
    """
    precision_map = {}
    processed_data = {}
    for timestamp, operation in data.items():
        for direction in ["paid", "received"]:
            if operation[direction]["asset_id"] not in precision_map:
                precision_map[operation[direction]["asset_id"]] = precision(
                    rpc, operation[direction]["asset_id"]
                )
        processed_data[timestamp] = {
            **{
                direction: {
                    "amount": (
                        float(operation[direction]["amount"])
                        / 10 ** precision_map[operation[direction]["asset_id"]]
                    ),
                    "asset_id": operation[direction]["asset_id"],
                }
                for direction in ["paid", "received"]
            },
            "account": operation["account"],
            "blocknum": operation["blocknum"],
            "op_id": operation["op_id"],
        }
    return processed_data


def parse_price_history(rpc, data):
    """
    Parse the price history of assets in a pool by processing the raw history data,
    converting amounts to float and calculating prices based on the asset pair.
    Parameters:
        data (dict): The raw history data to be processed.
    Returns:
        list: A list of lists containing
        timestamp, price, amount, account, blocknum, and op_id.
    """
    data = format_raw_history_data(data)
    data = process_operations_history(rpc, data)
    assets = [
        list(data.values())[0]["paid"]["asset_id"],
        list(data.values())[0]["received"]["asset_id"],
    ]
    assets = [int(i[4:]) for i in assets]
    assets.sort()
    assets = [f"1.3.{str(i)}" for i in assets]
    data2 = []
    for timestamp, operation in data.items():
        price = (
            operation["paid"]["amount"] / operation["received"]["amount"]
            if (
                operation["paid"]["asset_id"] == assets[0]
                and operation["received"]["asset_id"] == assets[1]
            )
            else (
                operation["received"]["amount"] / operation["paid"]["amount"]
                if (
                    operation["paid"]["asset_id"] == assets[1]
                    and operation["received"]["asset_id"] == assets[0]
                )
                else 0
            )
        )
        if price:
            data2.append(
                [
                    timestamp,
                    price,
                    operation["received"]["amount"],
                    operation["account"],
                    operation["blocknum"],
                    operation["op_id"],
                ]
            )
        else:
            print(timestamp, operation)
    return data2


def discrete_to_candles(discrete, size):
    """
    Converts a list of discrete data points into candle data.
    Args:
    - discrete (list): A list of discrete data points in the format
    [[unix_timestamp, price, volume], ...].
    - size (int): The size of the candles to create in seconds.
    Returns:
    - list: A list of candle data in the format
    [{"high": float, "low": float, "open": float,
    "close": float, "unix": int, "volume": float}, ...].
    """
    # Create a dictionary to store the discrete data points
    buckets = {}
    # Calculate the start and stop times for the candles
    start = int(size * (min(d[0] for d in discrete) // size))
    stop = int(size * (max(d[0] for d in discrete) // size))
    # Create a list of breaks to divide the data into candles
    breaks = list(range(start - 2 * size, stop + 2 * size, size))
    # Group the discrete data points into the buckets dictionary
    for event in discrete:
        # Find the correct bucket for the event
        bucket = breaks[bisect(breaks, event[0])]
        # Add the event to the correct bucket
        buckets.setdefault(bucket, []).append(event)
    # Convert the buckets into candle data
    return [
        {
            "high": max(d[1] for d in data),
            "low": min(d[1] for d in data),
            "open": data[0][1],
            "close": data[-1][1],
            "unix": bucket,
            "volume": sum(i[2] for i in data),
        }
        for bucket, data in buckets.items()
    ]


def update_candles_from_discrete(data, interval):
    """
    This function updates the historical candlestick data for various time intervals.
    Parameters:
    data (dict): A dictionary of dictionaries with the following keys:
        ["discrete", "c900", "c1800", "c3600", "c7200", "c14400", "c43200", "c86400"]
        The "discrete" key holds a list of discrete data points
        of the format [unix, price, volume] that go beyond the other data points.
        The other keys hold historical candlestick data of the given interval,
        that do not have the latest data.
        Each candlestick is a dictionary with keys:
        ["open", "high", "low", "close", "volume", "unix"]
    Returns:
    dict: The updated dictionary of historical candlestick data.
    """
    period = str(interval)
    if not data:
        candles = []
    else:
        candles = make_candles(list(zip(*data)), interval)
    candles_by_unix = {i["unix"]: i for i in candles}
    candles_by_unix = {i: candles_by_unix[i] for i in sorted(candles_by_unix)}
    return candles_by_unix


def get_discrete_data(rpc, market, start):
    """
    This function retrieves recent trade history for a given market,
    appends the latest data to an existing data set
    :param market: The market to retrieve trade history for
    :type market: str
    :return: The updated data set with recent trade history and formatted for display
    :rtype: dict
    """
    stop = time.time()
    startel = time.time()
    # retrieve recent trade history
    recent_data = get_trade_history(rpc, market, start, stop)
    print(time.time() - startel)
    # unzip and add the recent data to the existing data set
    discrete = recent_data
    discrete = sorted(discrete, key=lambda x: x[0])
    discrete = [list(i) for i in zip(*discrete)]
    return discrete, stop


def fetch_candles(rpc, market, start, stop, candle_size):
    """
    Fetches candles for a given market, with a given chart type, from the SQL database.
    Args:
        rpc (object): The RPC client instance.
        market (str): The market in the format of "<base_symbol>:<quote_symbol>"
        chart_type (str): The chart type that specifies how the candle data is formatted
    Returns:
        dict: The candle data, with keys are period and values being the list of data.
    """
    # Update the candle data in the database
    candles, stop = get_discrete_data(rpc, market, start)
    candles = update_candles_from_discrete(candles, candle_size)

    return list(candles.values())


def demo():
    rpc = wss_handshake()
    candles = fetch_candles(
        rpc, "1.19.160", time.time() - 86400 * 90, time.time(), 86400
    )

    # Prepare the data
    data = {
        "unix": [datetime.datetime.fromtimestamp(i["unix"]) for i in candles],
        "high": [i["high"] for i in candles],
        "low": [i["low"] for i in candles],
        "open": [i["open"] for i in candles],
        "close": [i["close"] for i in candles],
    }

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, data["high"], color="red", label="High")
    plt.plot(dates, data["low"], color="green", label="Low")
    plt.plot(dates, data["open"], color="orange", label="Open")
    plt.plot(dates, data["close"], color="yellow", label="Close")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Candlestick Data Over Time")
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    demo()
