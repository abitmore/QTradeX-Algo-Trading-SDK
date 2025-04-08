from math import ceil, inf

import matplotlib.pyplot as plt


class BaseBot:
    def autorange(self):
        """
        Returns:
         - An integer number of days that this bot requires to warm up its indicators
        """
        return ceil(max(v for k, v in self.tune.items() if k.endswith("_period")))

    def indicators(self, data):
        raise NotImplementedError

    def plot(self, data, states, indicators, block):
        raise NotImplementedError

    def strategy(self, state, indicators):
        raise NotImplementedError

    def reset(self):
        """
        reset any internal storage classes
        to be implemented by user
        """
        pass

    def execution(self, signal, indicators, wallet):
        return signal

    def fitness(self, states, raw_states, asset, currency):
        return ["roi"], {}
