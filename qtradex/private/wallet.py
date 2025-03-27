from copy import deepcopy


class WalletBase:
    def __init__(self, exchange, account):
        self.balances = {}
        self.exchange = exchange
        self.account = account
        self._readonly = True

    def __repr__(self):
        return f"{type(self)}{self.balances}"

    def __getitem__(self, index):
        # print("getitem", index)
        return self.balances[index]

    def __setitem__(self, index, item):
        # print("setitem", index, item, self._readonly)
        if not self._readonly:
            self.balances[index] = item

    def items(self):
        return self.balances.items()

    def keys(self):
        return self.balances.keys()

    def values(self):
        return self.balances.values()

    def copy(self):
        # create a new instance of a given subclass
        new_wallet = type(self)()
        new_wallet._readonly = self._readonly
        new_wallet.balances = self.balances.copy()
        return new_wallet

    def value(self, pair, price=None):
        if price is None:
            price = self.price
        else:
            self.price = price
        # return (
        #     (self.balances[pair[0]] + self.balances[pair[1]] / price)
        #     * (self.balances[pair[0]] * price + self.balances[pair[1]])
        # ) ** 0.5
        # wolfram alpha simplified:
        return (
            (self.balances[pair[0]] * price + self.balances[pair[1]]) ** 2 / price
        ) ** 0.5


class PaperWallet(WalletBase):
    def __init__(self, balances=None):
        super().__init__("paper", "nullaccount")
        self.balances = balances if balances is not None else {}
        self._readonly = False

    def _protect(self):
        self._readonly = True

    def _release(self):
        self._readonly = False


class Wallet(WalletBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FIXME add account monitoring & signing hooks so this is a live wallet
