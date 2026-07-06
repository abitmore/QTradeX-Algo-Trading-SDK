from qtradex.indicators.cache_decorator import cache, float_period
import tulipy


@cache
def abs(*args, **kwargs):
    return tulipy.abs(*args, **kwargs)


@cache
def acos(*args, **kwargs):
    return tulipy.acos(*args, **kwargs)


@cache
def ad(*args, **kwargs):
    return tulipy.ad(*args, **kwargs)


@cache
def add(*args, **kwargs):
    return tulipy.add(*args, **kwargs)


@float_period(4, 5)
@cache
def adosc(*args, **kwargs):
    return tulipy.adosc(*args, **kwargs)


@float_period(3,)
@cache
def adx(*args, **kwargs):
    return tulipy.adx(*args, **kwargs)


@float_period(3,)
@cache
def adxr(*args, **kwargs):
    return tulipy.adxr(*args, **kwargs)


@cache
def ao(*args, **kwargs):
    return tulipy.ao(*args, **kwargs)


@float_period(1, 2)
@cache
def apo(*args, **kwargs):
    return tulipy.apo(*args, **kwargs)


@float_period(2,)
@cache
def aroon(*args, **kwargs):
    return tulipy.aroon(*args, **kwargs)


@float_period(2,)
@cache
def aroonosc(*args, **kwargs):
    return tulipy.aroonosc(*args, **kwargs)


@cache
def asin(*args, **kwargs):
    return tulipy.asin(*args, **kwargs)


@cache
def atan(*args, **kwargs):
    return tulipy.atan(*args, **kwargs)


@float_period(3,)
@cache
def atr(*args, **kwargs):
    return tulipy.atr(*args, **kwargs)


@cache
def avgprice(*args, **kwargs):
    return tulipy.avgprice(*args, **kwargs)


@float_period(1,)
@cache
def bbands(*args, **kwargs):
    return tulipy.bbands(*args, **kwargs)


@cache
def bop(*args, **kwargs):
    return tulipy.bop(*args, **kwargs)


@float_period(3,)
@cache
def cci(*args, **kwargs):
    return tulipy.cci(*args, **kwargs)


@cache
def ceil(*args, **kwargs):
    return tulipy.ceil(*args, **kwargs)


@float_period(1,)
@cache
def cmo(*args, **kwargs):
    return tulipy.cmo(*args, **kwargs)


@cache
def cos(*args, **kwargs):
    return tulipy.cos(*args, **kwargs)


@cache
def cosh(*args, **kwargs):
    return tulipy.cosh(*args, **kwargs)


@cache
def crossany(*args, **kwargs):
    return tulipy.crossany(*args, **kwargs)


@cache
def crossover(*args, **kwargs):
    return tulipy.crossover(*args, **kwargs)


@float_period(2,)
@cache
def cvi(*args, **kwargs):
    return tulipy.cvi(*args, **kwargs)


@float_period(1,)
@cache
def decay(*args, **kwargs):
    return tulipy.decay(*args, **kwargs)


@float_period(1,)
@cache
def dema(*args, **kwargs):
    return tulipy.dema(*args, **kwargs)


@float_period(3,)
@cache
def di(*args, **kwargs):
    return tulipy.di(*args, **kwargs)


@cache
def div(*args, **kwargs):
    return tulipy.div(*args, **kwargs)


@float_period(2,)
@cache
def dm(*args, **kwargs):
    return tulipy.dm(*args, **kwargs)


@float_period(1,)
@cache
def dpo(*args, **kwargs):
    return tulipy.dpo(*args, **kwargs)


@float_period(3,)
@cache
def dx(*args, **kwargs):
    return tulipy.dx(*args, **kwargs)


@float_period(1,)
@cache
def edecay(*args, **kwargs):
    return tulipy.edecay(*args, **kwargs)


@float_period(1,)
@cache
def ema(*args, **kwargs):
    return tulipy.ema(*args, **kwargs)


@cache
def emv(*args, **kwargs):
    return tulipy.emv(*args, **kwargs)


@cache
def exp(*args, **kwargs):
    return tulipy.exp(*args, **kwargs)


@float_period(2,)
@cache
def fisher(*args, **kwargs):
    return tulipy.fisher(*args, **kwargs)


@cache
def floor(*args, **kwargs):
    return tulipy.floor(*args, **kwargs)


@float_period(1,)
@cache
def fosc(*args, **kwargs):
    return tulipy.fosc(*args, **kwargs)


@float_period(1,)
@cache
def hma(*args, **kwargs):
    return tulipy.hma(*args, **kwargs)


@float_period(1,)
@cache
def kama(*args, **kwargs):
    return tulipy.kama(*args, **kwargs)


@float_period(4, 5)
@cache
def kvo(*args, **kwargs):
    return tulipy.kvo(*args, **kwargs)


@float_period(1,)
@cache
def lag(*args, **kwargs):
    return tulipy.lag(*args, **kwargs)


@float_period(1,)
@cache
def linreg(*args, **kwargs):
    return tulipy.linreg(*args, **kwargs)


@float_period(1,)
@cache
def linregintercept(*args, **kwargs):
    return tulipy.linregintercept(*args, **kwargs)


@float_period(1,)
@cache
def linregslope(*args, **kwargs):
    return tulipy.linregslope(*args, **kwargs)


@cache
def ln(*args, **kwargs):
    return tulipy.ln(*args, **kwargs)


@cache
def log10(*args, **kwargs):
    return tulipy.log10(*args, **kwargs)


@float_period(1, 2, 3)
@cache
def macd(*args, **kwargs):
    return tulipy.macd(*args, **kwargs)


@cache
def marketfi(*args, **kwargs):
    return tulipy.marketfi(*args, **kwargs)


@float_period(2,)
@cache
def mass(*args, **kwargs):
    return tulipy.mass(*args, **kwargs)


@float_period(1,)
@cache
def max(*args, **kwargs):
    return tulipy.max(*args, **kwargs)


@float_period(1,)
@cache
def md(*args, **kwargs):
    return tulipy.md(*args, **kwargs)


@cache
def medprice(*args, **kwargs):
    return tulipy.medprice(*args, **kwargs)


@float_period(4,)
@cache
def mfi(*args, **kwargs):
    return tulipy.mfi(*args, **kwargs)


@float_period(1,)
@cache
def min(*args, **kwargs):
    return tulipy.min(*args, **kwargs)


@float_period(1,)
@cache
def mom(*args, **kwargs):
    return tulipy.mom(*args, **kwargs)


@float_period(1,)
@cache
def msw(*args, **kwargs):
    return tulipy.msw(*args, **kwargs)


@cache
def mul(*args, **kwargs):
    return tulipy.mul(*args, **kwargs)


@float_period(3,)
@cache
def natr(*args, **kwargs):
    return tulipy.natr(*args, **kwargs)


@cache
def nvi(*args, **kwargs):
    return tulipy.nvi(*args, **kwargs)


@cache
def obv(*args, **kwargs):
    return tulipy.obv(*args, **kwargs)


@float_period(1, 2)
@cache
def ppo(*args, **kwargs):
    return tulipy.ppo(*args, **kwargs)


@cache
def psar(*args, **kwargs):
    return tulipy.psar(*args, **kwargs)


@cache
def pvi(*args, **kwargs):
    return tulipy.pvi(*args, **kwargs)


@float_period(2,)
@cache
def qstick(*args, **kwargs):
    return tulipy.qstick(*args, **kwargs)


@float_period(1,)
@cache
def roc(*args, **kwargs):
    return tulipy.roc(*args, **kwargs)


@float_period(1,)
@cache
def rocr(*args, **kwargs):
    return tulipy.rocr(*args, **kwargs)


@cache
def round(*args, **kwargs):
    return tulipy.round(*args, **kwargs)


@float_period(1,)
@cache
def rsi(*args, **kwargs):
    return tulipy.rsi(*args, **kwargs)


@cache
def sin(*args, **kwargs):
    return tulipy.sin(*args, **kwargs)


@cache
def sinh(*args, **kwargs):
    return tulipy.sinh(*args, **kwargs)


@float_period(1,)
@cache
def sma(*args, **kwargs):
    return tulipy.sma(*args, **kwargs)


@cache
def sqrt(*args, **kwargs):
    return tulipy.sqrt(*args, **kwargs)


@float_period(1,)
@cache
def stddev(*args, **kwargs):
    return tulipy.stddev(*args, **kwargs)


@float_period(1,)
@cache
def stderr(*args, **kwargs):
    return tulipy.stderr(*args, **kwargs)


@float_period(3, 4, 5)
@cache
def stoch(*args, **kwargs):
    return tulipy.stoch(*args, **kwargs)


@float_period(1,)
@cache
def stochrsi(*args, **kwargs):
    return tulipy.stochrsi(*args, **kwargs)


@cache
def sub(*args, **kwargs):
    return tulipy.sub(*args, **kwargs)


@float_period(1,)
@cache
def sum(*args, **kwargs):
    return tulipy.sum(*args, **kwargs)


@cache
def tan(*args, **kwargs):
    return tulipy.tan(*args, **kwargs)


@cache
def tanh(*args, **kwargs):
    return tulipy.tanh(*args, **kwargs)


@float_period(1,)
@cache
def tema(*args, **kwargs):
    return tulipy.tema(*args, **kwargs)


@cache
def todeg(*args, **kwargs):
    return tulipy.todeg(*args, **kwargs)


@cache
def torad(*args, **kwargs):
    return tulipy.torad(*args, **kwargs)


@cache
def tr(*args, **kwargs):
    return tulipy.tr(*args, **kwargs)


@float_period(1,)
@cache
def trima(*args, **kwargs):
    return tulipy.trima(*args, **kwargs)


@float_period(1,)
@cache
def trix(*args, **kwargs):
    return tulipy.trix(*args, **kwargs)


@cache
def trunc(*args, **kwargs):
    return tulipy.trunc(*args, **kwargs)


@float_period(1,)
@cache
def tsf(*args, **kwargs):
    return tulipy.tsf(*args, **kwargs)


@cache
def typprice(*args, **kwargs):
    return tulipy.typprice(*args, **kwargs)


@float_period(3, 4, 5)
@cache
def ultosc(*args, **kwargs):
    return tulipy.ultosc(*args, **kwargs)


@float_period(1,)
@cache
def var(*args, **kwargs):
    return tulipy.var(*args, **kwargs)


@float_period(1,)
@cache
def vhf(*args, **kwargs):
    return tulipy.vhf(*args, **kwargs)


@float_period(1, 2)
@cache
def vidya(*args, **kwargs):
    return tulipy.vidya(*args, **kwargs)


@float_period(1,)
@cache
def volatility(*args, **kwargs):
    return tulipy.volatility(*args, **kwargs)


@float_period(1, 2)
@cache
def vosc(*args, **kwargs):
    return tulipy.vosc(*args, **kwargs)


@float_period(2,)
@cache
def vwma(*args, **kwargs):
    return tulipy.vwma(*args, **kwargs)


@cache
def wad(*args, **kwargs):
    return tulipy.wad(*args, **kwargs)


@cache
def wcprice(*args, **kwargs):
    return tulipy.wcprice(*args, **kwargs)


@float_period(1,)
@cache
def wilders(*args, **kwargs):
    return tulipy.wilders(*args, **kwargs)


@float_period(3,)
@cache
def willr(*args, **kwargs):
    return tulipy.willr(*args, **kwargs)


@float_period(1,)
@cache
def wma(*args, **kwargs):
    return tulipy.wma(*args, **kwargs)


@float_period(1,)
@cache
def zlema(*args, **kwargs):
    return tulipy.zlema(*args, **kwargs)
