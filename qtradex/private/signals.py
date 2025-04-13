import math


class Buy:
    def __init__(self, price=None, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True


class Sell:
    def __init__(self, price=None, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True


class Thresholds:
    def __init__(self, buying, selling, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = None
        self.unix = 0
        self.profit = 0
        self.buying = buying
        self.selling = selling
