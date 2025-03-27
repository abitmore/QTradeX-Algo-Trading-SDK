import math


class Buy:
    def __init__(self, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = float("nan")
        self.unix = 0
        self.profit = 0


class Sell:
    def __init__(self, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = float("nan")
        self.unix = 0
        self.profit = 0


class Thresholds:
    def __init__(self, buying, selling, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = float("nan")
        self.unix = 0
        self.profit = 0
        self.buying = buying
        self.selling = selling
