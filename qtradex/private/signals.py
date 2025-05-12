import math


class SignalBase:
    def __repr__(self):
        return f"{type(self)}(profit={self.profit}, price={self.price}, unix={self.unix})"

class Buy(SignalBase):
    def __init__(self, price=None, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True


class Sell(SignalBase):
    def __init__(self, price=None, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True


class Thresholds(SignalBase):
    def __init__(self, buying, selling, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = None
        self.unix = 0
        self.profit = 0
        self.buying = buying
        self.selling = selling


class Hold(SignalBase):
    """
    AKA Cancel All Orders
    """
