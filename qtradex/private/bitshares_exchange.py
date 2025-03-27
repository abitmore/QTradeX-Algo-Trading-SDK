import bitshares_signing.rpc as bitshares_rpc
from bitshares_signing import broker, prototype_order
from bitshares_signing.config import NODES


class BitsharesExchange:
    def __init__(self, name, wif):
        self.rpc = bitshares_rpc.wss_handshake()
        self.account_name = name
        self.account_id = bitshares_rpc.rpc_get_account(self.rpc, self.account_name)
        self.wif = wif

        self.login()

    def _prototype(self, symbol):
        asset, currency = symbol.split("/")
        order = prototype_order(
            {
                "asset_id": bitshares_rpc.id_from_name(asset),
                "asset_precision": bitshares_rpc.precision(asset),
                "currency_id": bitshares_rpc.id_from_name(currency),
                "currency_precision": bitshares_rpc.precision(currency),
                "account_id": self.account_id,
                "account_name": self.account_name,
                "wif": self.wif,
            }
        )
        return order

    def login(self):
        rpc_name = bitshares_rpc.get_objects(rpc, self.account_id)
        assert (
            self.account_name == rpc_name
        ), f'Invalid account name! "{self.account_name}" does not exist on BitShares.'
        order = {
            "edicts": [{"op": "login"}],
            "header": {
                "account_id": self.account_id,
                "account_name": self.account_name,
                "wif": self.wif,
            },
            "nodes": NODES,
        }
        assert broker(order), "Failed to authenticate!"

    def create_order(self, symbol, order_type, side, amount, price):
        order = self._prototype(symbol)

        if order_type == "swap":
            pass
            # FIXME implement swaps
        if order_type == "limit":
            assert side in ["buy", "sell"]
            order["edicts"].append({"op": side, "amount": amount, "price": price})
            return broker(order)
            # FIXME order id callback
        else:
            raise ValueError(f"Invalid order_type {order_type}")



    def cancel_order(self, order_id, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": [order_id]})
        return broker(order)

    def cancel_orders(self, ids, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": ids})
        return broker(order)

    def cancel_all_orders(self, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": ["1.7.X"]})
        return broker(order)

    def fetch_open_order(self, order_id, _):
        return bitshares_rpc.rpc_get_objects(self.rpc, order_id)

    def fetch_open_orders(self, symbol):
        pair = self._prototype(symbol)["header"]
        return bitshares_rpc.rpc_open_orders(self.rpc, self.account_name, pair)

    def fetch_balance(self, _):
        return bitshares_rpc.rpc_balances(self.rpc, self.account_name)

    def fetch_ticker(self, symbol):
        """
        Returns:
            {
                "bid": float(),
                "ask": float(),
            }
        """
        pair = self._prototype(symbol)["header"]
        return bitshares_rpc.rpc_ticker(self.rpc, pair["asset_id"], pair["currency_id"])
