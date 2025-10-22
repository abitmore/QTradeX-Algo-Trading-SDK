from math import ceil, inf

import matplotlib.pyplot as plt


class BaseBot:
    def autorange(self):
        """
        Returns:
         - An integer number of days that this bot requires to warm up its indicators
        """
        return (
            ceil(max(v for k, v in self.tune.items() if k.endswith("_period")))
            if any(k.endswith("_period") for k in self.tune)
            else 0
        )

    def indicators(self, data):
        raise NotImplementedError

    def plot(self, data, states, indicators, block):
        axes = qx.plot(
            self.info,
            data,
            states,
            indicators,
            block,
            tuple(),
        )

    def strategy(self, state, indicators):
        return None

    def reset(self):
        """
        reset any internal storage classes
        to be implemented by user
        """
        pass

    def execution(self, signal, indicators, wallet):
        return signal

    def fitness(self, states, raw_states, asset, currency):
        return ["roi", "cagr", "trade_win_rate"], {}


class Info(dict):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        raise TypeError(
            "This dictionary is read-only. Use the '_set' method to update values."
        )

    def _set(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return repr(self._data)

    def __contains__(self, key):
        return key in self._data
