import json
from typing import Dict, List, Union

import numpy as np
import qtradex as qx
from qtradex.common.utilities import rotate
from qtradex.core.quant import preprocess_states, slice_candles
from qtradex.private.signals import Buy, Sell, Thresholds
from qtradex.private.wallet import PaperWallet


def trade(asset, currency, operation, wallet, price, now):
    initial_value = wallet.value((asset, currency))

    execution = price["close"]
    if isinstance(operation, Thresholds):
        if wallet[asset]:
            if price["high"] > operation.selling:
                execution = operation.selling
                operation = Sell(maxvolume=operation.maxvolume)
        elif wallet[currency]:
            if price["low"] < operation.buying:
                execution = operation.buying
                operation = Buy(maxvolume=operation.maxvolume)
    elif operation.price is not None:
        if wallet[asset] and isinstance(operation, Sell):
            if price["high"] > operation.price:
                execution = operation.price
            else:
                operation = None
        elif wallet[currency] and isinstance(operation, Buy):
            if price["low"] < operation.price:
                execution = operation.price
            else:
                operation = None

    execution = min(max(execution, price["low"]), price["high"])

    # print(execution, type(operation))

    if isinstance(operation, Buy):
        volume = min(wallet[currency], operation.maxvolume)
        if not volume:
            return wallet, None
        wallet[asset] = wallet[asset] + volume / execution
        wallet[currency] -= volume

    elif isinstance(operation, Sell):
        volume = min(wallet[asset], operation.maxvolume)
        if not volume:
            return wallet, None
        wallet[asset] -= volume
        wallet[currency] = wallet[currency] + volume * execution

    if isinstance(operation, (Buy, Sell)):
        operation.price = execution
        operation.unix = now
        operation.profit = wallet.value((asset, currency), execution) / initial_value
    else:
        operation = None

    return wallet, operation


def backtest(bot, data, wallet=None, plot=True, block=True, return_states=False, range_periods=True, show=True):
    if wallet is None:
        wallet = PaperWallet({data.asset:0, data.currency:1})
    bot.reset()
    begin = data.begin
    end = data.end
    days = (end - begin) / 86400
    candle_size = data.candle_size
    warmup = bot.autorange()

    orig_tune = bot.tune.copy()

    if range_periods:
        # allow for different candle sizes whilst maintaining the type of the parameter
        for k, v in list(bot.tune.items()):
            if k.endswith("_period"):
                if isinstance(v, float):
                    bot.tune[k] = v * (86400 / candle_size)
                else:
                    bot.tune[k] = int(v * (86400 / candle_size))


    now = begin + (candle_size * (warmup + 1))

    initial_data = slice_candles(now, data, candle_size, 1)

    wallet.value((data.asset, data.currency), initial_data["close"])

    indicator_states = []
    states = []

    indicators = bot.indicators(data)

    minlen = min(map(len, indicators.values()))

    # print({k: type(v) for k, v in indicators.items()})
    indicators = {k: v[-minlen:] for k, v in indicators.items()}

    # offset = len(data["unix"]) - len(indicators)

    indicated_data = {"indicators": rotate(indicators)}
    indicated_data.update({k: v[-minlen:] for k, v in data.items()})
    last_trade = None

    if now > end:
        states.append(
            {"trades": None, "balances": wallet.copy(), "unix": now, **initial_data}
        )

    last_trade_time = 0

    while now <= end:
        tickdx = np.searchsorted(indicated_data["unix"], now, side="left")
        try:
            tick_data = {k: v[tickdx] for k, v in indicated_data.items()}
        except:
            now += candle_size
            # print("fast forward", begin, candle_size, now, end, tickdx)
            continue

        # make the wallet read-only before passing it to the user
        # this is not fail-safe, just helpful to keep accidental modifications out
        wallet._protect()
        indicators = tick_data["indicators"]
        operation = bot.strategy(
            {"last_trade": last_trade, "unix": now, "wallet": wallet, **tick_data},
            indicators,
        )
        # keep the last trade
        if operation is not None:
            last_trade = operation

        # if enough time has elapsed and we can trade again
        if now - last_trade_time >= data.base_size and operation is not None:
            # release write protection and trade
            wallet._release()
            wallet, operation = trade(
                data.asset, data.currency, operation, wallet, tick_data, now
            )
            last_trade_time = now
        else:
            # this one doesn't count
            operation = None
        states.append(
            {
                "trades": operation,
                "balances": wallet.copy(),
                "unix": now,
                **tick_data,
            }
        )
        indicator_states.append(indicators)
        now += candle_size

    states = rotate(states)
    states["trades"] = [i for i in states["trades"] if i is not None]
    indicator_states = rotate(indicator_states)

    if states["trades"]:
        states["trade_times"], states["trade_prices"] = list(
            zip(*[[op.unix, op.price] for op in states["trades"]])
        )
    else:
        states["trade_times"], states["trade_prices"] = [], []
    states["trade_colors"] = [
        "green" if isinstance(i, Buy) else "red" for i in states["trades"]
    ]

    raw_states = states
    states = preprocess_states(states, (data.asset, data.currency))

    states["days"] = days

    keys, custom = bot.fitness(states, raw_states, data.asset, data.currency)
    if "roi" not in keys:
        keys.append("roi")

    bot.tune = orig_tune

    ret = {
        **qx.indicators.fitness.fitness(
            keys, states, raw_states, data.asset, data.currency
        ),
        **custom,
    }

    if plot:
        if show:
            print(json.dumps(ret, indent=4))
        bot.plot(data, raw_states, indicator_states, block)

    # if requested, return the raw states along with the fitness.
    # This is so that papertrade and live can see if the bot is currently buying or
    # selling even if the bot doesn't say to do anything on a given tick.
    if return_states:
        ret = [ret, raw_states, states]

    return ret
