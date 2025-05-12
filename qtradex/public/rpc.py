"""
╔═╗─┐ ┬┌┬┐┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌╔═╗┬  ┬┌─┐┌┐┌┌┬┐
║╣ ┌┴┬┘ │ │││││   │ ││ ││││║╣ └┐┌┘├┤ │││ │ 
╚═╝┴ └─ ┴ ┴┘└┘└─┘ ┴ ┴└─┘┘└┘╚═╝ └┘ └─┘┘└┘ ┴ 
"""

from json import dumps as json_dumps
from json import loads as json_loads

from qtradex.common.bitshares_nodes import bitshares_nodes
from qtradex.common.utilities import it, race_read, race_write, to_iso_date, trace
from websocket import create_connection as wss


def wss_handshake(node=bitshares_nodes):
    """
    create a new websocket connection
    """
    rpc = None
    if isinstance(node, list):
        for n in node:
            try:
                rpc = wss(n, timeout=5)
                if rpc is None:
                    continue
                break
            except:
                pass
        if rpc is None:
            raise ConnectionError(it("red", "Unable to connect to RPC, check nodes."))
    else:
        rpc = wss(node, timeout=5)
    return rpc


def wss_query(rpc, params):
    """
    this definition will place all remote procedure calls (RPC)
    """

    for _ in range(10):
        try:
            # this is the 4 part format of EVERY self.connection request
            # params format is ["location", "object", []]

            query = json_dumps(
                {"method": "call", "params": params, "jsonrpc": "2.0", "id": 1}
            )
            # self.connection is the websocket connection created by wss_handshake()
            # we will use this connection to send query and receive json
            rpc.send(query)
            ret = json_loads(rpc.recv())
            try:
                return ret["result"]  # if there is result key take it
            except Exception:
                print("NODE FAILED", params)
                return ret
        except Exception as error:
            try:  # attempt to terminate the connection
                rpc.close()
            except Exception:
                pass
            trace(error)
            rpc = wss_handshake()


def rpc_get_objects(rpc, obj_id):
    if isinstance(obj_id, list):
        if len(obj_id) > 49:
            ret = []
            for idx in range(0, obj_id - 49, 49):
                ret.extend(
                    wss_query(
                        rpc, ["database", "get_objects", [obj_id[idx : idx + 49]]]
                    )
                )
        else:
            ret = wss_query(rpc, ["database", "get_objects", [obj_id]])
    else:
        ret = wss_query(rpc, ["database", "get_objects", [[obj_id]]])[0]
    return ret


def rpc_get_multiple_objects(rpc, assets):
    return wss_query(rpc, ["database", "get_objects", [assets]])


def rpc_market_history(rpc, currency_id, asset_id, period, start, stop):
    """
    kline data
    """
    return wss_query(
        rpc,
        [
            "history",
            "get_market_history",
            [currency_id, asset_id, period, to_iso_date(start), to_iso_date(stop)],
        ],
    )


def rpc_lookup_asset_symbols(rpc, asset, currency):
    """
    given asset name return 1.3.x
    """
    return wss_query(rpc, ["database", "lookup_asset_symbols", [[asset, currency]]])


def precision(rpc, object_id):
    try:
        with open("precisions.txt") as handle:
            precs = json_loads(handle.read())
            handle.close()
    except FileNotFoundError:
        precs = {}
    if isinstance(object_id, str):
        if object_id in precs:
            return precs[object_id]
        prec = int(rpc_get_objects(rpc, object_id)["precision"])
        # print(prec)
        precs[object_id] = prec
        with open("precisions.txt", "w") as handle:
            handle.write(json_dumps(precs))
            handle.close()
        return prec
    else:
        print(object_id)
        raise ValueError()
        # availible = {obid:precs[obid] for obid in object_id if obid in precs}
        # rpc_ids = [obid for obid in object_id if obid not in availible]
        rpc_precs = {
            rpc_ids[idx]: i["precision"]
            for idx, i in enumerate(rpc_get_multiple_objects(rpc, object_id))
        }
        precs = {**precs, **rpc_precs}
        with open("precisions.txt", "w") as handle:
            handle.write(json_dumps(precs))
            handle.close()
        return list(rpc_precs.values())


def id_from_name(rpc, object_name):
    try:
        precs = race_read("ids_to_names.txt")
    except FileNotFoundError:
        precs = {}
    if object_name in precs:
        return precs[object_name]
    prec = rpc_lookup_asset_symbols(rpc, object_name, object_name)[0]["id"]
    precs[object_name] = prec
    race_write("ids_to_names.txt", json_dumps(precs))
    return prec


def id_to_name(rpc, object_id):
    try:
        precs = race_read("names_to_ids.txt")
    except FileNotFoundError:
        precs = {}
    if object_id in precs:
        return precs[object_id]
    prec = rpc_get_objects(rpc, object_id)["symbol"]
    precs[object_id] = prec
    race_write("names_to_ids.txt", json_dumps(precs))
    return prec


def rpc_pool_last(rpc, pool, in_terms):
    getobj = rpc_get_objects(rpc, pool)
    bal_a = int(getobj["balance_a"]) / 10 ** precision(rpc, getobj["asset_a"])
    bal_b = int(getobj["balance_b"]) / 10 ** precision(rpc, getobj["asset_b"])
    if in_terms == id_to_name(rpc, getobj["asset_a"]):
        return bal_a / bal_b
    else:
        return bal_b / bal_a


def rpc_account_by_name(rpc, account):
    return wss_query(rpc, ["database", "get_account_by_name", [account, 1]])


def rpc_book(rpc, asset, currency, depth=3):
    """
    Remote procedure call orderbook bids and asks
    ~
    :RPC param str(base): symbol name or ID of the base asset
    :RPC param str(quote): symbol name or ID of the quote asset
    :RPC param int(limit): depth of the order book to retrieve (max limit 50)
    :RPC returns: Order book of the market
    """
    asset = {"name": asset, "precision": precision(rpc, id_from_name(rpc, asset))}
    currency = {
        "name": currency,
        "precision": precision(rpc, id_from_name(rpc, currency)),
    }
    order_book = wss_query(
        rpc, ["database", "get_order_book", [currency["name"], asset["name"], depth]]
    )
    askp, askv, bidp, bidv = [], [], [], []
    for i, _ in enumerate(order_book["asks"]):
        price = float(order_book["asks"][i]["price"]) / 10**16
        if float(price) == 0:
            raise ValueError("zero price in asks")
        volume = float(order_book["asks"][i]["quote"]) / 10 ** int(asset["precision"])
        askp.append(price)
        askv.append(volume)
    for i, _ in enumerate(order_book["bids"]):
        price = float(order_book["bids"][i]["price"]) / 10**16
        if float(price) == 0:
            raise ValueError("zero price in bids")
        volume = float(order_book["bids"][i]["quote"]) / 10 ** int(asset["precision"])
        bidp.append(price)
        bidv.append(volume)
    return {"askp": askp, "askv": askv, "bidp": bidp, "bidv": bidv}


def rpc_last(rpc, pair):
    """
    RPC the latest ticker price
    ~
    :RPC param base: symbol name or ID of the base asset
    :RPC param quote: symbol name or ID of the quote asset
    :RPC returns: The market ticker for the past 24 hours
    """
    asset, currency = pair.split(":")
    ticker = wss_query(rpc, ["database", "get_ticker", [currency, asset, False]])
    last = float(ticker["latest"]) / 10**16
    if float(last) == 0:
        raise ValueError("zero price last")
    return last


def get_bitshares_balances(rpc, api):
    asset, currency = api["pair"].split(":")
    asset, currency = id_from_name(rpc, asset), id_from_name(rpc, currency)
    balances = wss_query(
        rpc,
        ["database", "get_named_account_balances", [api["user_id"], [asset, currency]]],
    )
    idx = [i["asset_id"] for i in balances].index(asset)
    # FIXME TESTING
    return {"asset_free": 1, "currency_free": 1000}
    return {
        "asset_free": int(balances[idx]["amount"]) / 10 ** precision(rpc, asset),
        "currency_free": int(balances[-idx + 1]["amount"])
        / 10 ** precision(rpc, currency),
    }


def rpc_pool_book(pool_data):
    # async def gather_orderbook(self, pool_data, rpc, pair, req_params, ws):
    """
    Gather orderbook information either from the pool data or via RPC request
    Parameters:
        pool_data (dict): The data of the pool
        rpc (object): The rpc instance to be used to make the request
        pair (str): The asset pair being queried
        req_params (dict): The request parameters used gather the orderbook

    Returns:
        data (dict): A dictionary containing the bid and ask orderbook information
    """

    def pool(x_start, y_start, delta_x):
        """
        x_start*y_start = k
        x1 = x_start+delta_x
        k / x1 = y1
        y1-y_start = delta_y
        """
        return y_start - (x_start * y_start) / (x_start + delta_x)

    balance_a = pool_data["balance_a"]
    balance_b = pool_data["balance_b"]
    asset = pool_data["asset"]
    currency = pool_data["currency"]
    konstant = balance_a * balance_b

    # List to store the order book
    bidp, bidv, askp, askv = [], [], [], []
    step = 0.01 * balance_a

    for i in range(1, 99):
        delta_a = i * step
        balance_a2 = balance_a + delta_a
        balance_b2 = konstant / balance_a2
        delta_b = abs(balance_b - balance_b2)
        price = delta_a / delta_b
        # gain = step * price
        askp.append(price)
        askv.append(step)

    for i in range(1, 99):
        delta_a = i * step
        balance_a2 = balance_a - delta_a
        balance_b2 = konstant / balance_a2
        delta_b = abs(balance_b - balance_b2)
        price = delta_a / delta_b
        # gain = step * price
        bidp.append(price)
        bidv.append(step)

    # Sort the order book by price
    askp, askv = map(list, zip(*sorted(zip(askp, askv), reverse=False)))
    bidp, bidv = map(list, zip(*sorted(zip(bidp, bidv), reverse=True)))
    # asks.sort(key=lambda x: x[0], reverse=False)
    # bids.sort(key=lambda x: x[0], reverse=True)

    # Print the order book
    # for order in order_book:
    #     print(f"Price: {order[0]:.8f}   Amount: {math.copysign(order[1], 1):.2f}")
    # data = {
    #     "bids": bids,
    #     "asks": asks,
    # }
    return {"askp": askp, "askv": askv, "bidp": bidp, "bidv": bidv}
    # return {"askp":[30000], "askv":[1], "bidp":[30000], "bidv":[1]}


def rpc_open_orders(rpc, account_name, storage, market):
    # return a list of open orders, for one account, in one market
    ret = wss_query(rpc, ["database", "get_full_accounts", [[account_name], "false"]])
    try:
        limit_orders = ret[0][1]["limit_orders"]
    except Exception:
        limit_orders = []
    orders = [
        order["id"]
        for order in limit_orders
        if (order["sell_price"]["base"]["asset_id"] in market)
        and (order["sell_price"]["quote"]["asset_id"] in market)
    ]
    orders = rpc_get_objects(rpc, orders)
    # print(orders)
    orders = [
        {
            "price": 1
            / (
                (
                    int(order["sell_price"]["base"]["amount"])
                    / 10 ** precision(rpc, order["sell_price"]["base"]["asset_id"])
                )
                / (
                    int(order["sell_price"]["quote"]["amount"])
                    / 10 ** precision(rpc, order["sell_price"]["quote"]["asset_id"])
                )
            ),
            "order_id": order["id"],
            "start_qty": int(order["sell_price"]["base"]["amount"])
            / 10 ** precision(rpc, order["sell_price"]["base"]["asset_id"]),
            "current_qty": int(order["filled_amount"])
            / 10 ** precision(rpc, order["sell_price"]["base"]["asset_id"]),
            "is_ask": order["sell_price"]["base"]["asset_id"] == market[0],
        }
        for order in orders
    ]
    print(orders)

    bids = [order for order in orders if not order["is_ask"]]
    asks = [order for order in orders if order["is_ask"]]

    orders = {
        "bids": bids,
        "asks": asks,
        "bid_sum": sum(i["current_qty"] for i in bids),
        "ask_sum": sum(i["current_qty"] for i in asks),
    }

    return orders
