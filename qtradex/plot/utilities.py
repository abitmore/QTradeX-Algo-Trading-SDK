from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from matplotlib.collections import LineCollection
from qtradex.common.utilities import NIL, expand_bools, rotate
from qtradex.private.signals import Buy, Sell


def plotmotion(block):
    if block:
        plt.ioff()
        plt.show()
    else:
        plt.ion()
        plt.pause(0.00001)


def plot_indicators(axes, timestamps, states, indicators, indicator_fmt):
    for key, name, color, idx, title in indicator_fmt:
        ax = axes[idx]
        ax.set_title(title)
        # Plot each EMA with a color gradient
        ax.plot(
            timestamps,
            indicators[key],
            color=color,
            label=name,
        )


def unix_to_stamp(unix):
    if isinstance(unix, (float, int)):
        return datetime.fromtimestamp(unix)
    else:
        return [datetime.fromtimestamp(i) for i in unix]


def plot(
    info,
    data,
    states,
    indicators,
    block,
    indicator_fmt,
    style="dark_background",
):
    """
    plotting of buy/sell with win/loss line plotting
    buy/sell are green/red triangles
    plotting of high/low/open/close
    plotting of indicators (dict of indicator keys to be plotted and color)
    balance plotting follows price on token not held

    During papertrade and live sessions, the plotting is a bit different.

    Notably:
    - the red and green `open - close` clouds are not displayed
    - an extra argument, `raw` is given as raw high-frequency data and the high/low for
      that is plotted instead
    - past live trades are passed in and no "backtest trades" are plotted past the earliest
    """
    mplstyle.use(style)

    n_levels = max(i[3] for i in indicator_fmt) + 2
    # clear the current figure
    plt.clf()
    axes = [plt.subplot(n_levels, 1, n) for n in range(1, n_levels + 1)]
    axes[0].set_yscale("log")

    timestamps = unix_to_stamp(states["unix"])
    states["dates"] = timestamps

    # plotting of high/low/open/close
    # high/low
    axes[0].fill_between(
        timestamps,
        states["low"],
        states["high"],
        color="magenta",
        alpha=0.3,
        label="High/Low",
    )
    if info["mode"] not in ["live", "papertrade"]:
        # Fill between for open > close
        axes[0].fill_between(
            timestamps,
            states["open"],
            states["close"],
            where=expand_bools(states["open"] > states["close"], side="right"),
            color=(1, 0, 0, 0.3),  # Red for open > close
        )

        # Fill between for open < close
        axes[0].fill_between(
            timestamps,
            states["open"],
            states["close"],
            where=expand_bools(states["open"] < states["close"], side="right"),
            color=(0, 1, 0, 0.3),  # Green for open < close
        )
    if "live_data" in info:
        high_res = info["live_data"]
        mindx = np.searchsorted(high_res["unix"], states["unix"][0], side="left")
        high_res = {k: v[-mindx:] for k, v in high_res.items()}

        # Fill between for open > close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["high"],
            high_res["low"],
            color=(1, 1, 1, 0.8),  # white green
            label="high > low",
        )

        # Fill between for open > close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["open"],
            high_res["close"],
            where=expand_bools(high_res["open"] > high_res["close"], side="right"),
            color=(1, 0.8, 0.8, 1),  # white red
            label="open > close",
        )

        # Fill between for open < close
        axes[0].fill_between(
            unix_to_stamp(high_res["unix"]),
            high_res["open"],
            high_res["close"],
            where=expand_bools(high_res["open"] < high_res["close"], side="right"),
            color=(0.8, 0.8, 1, 1),  # white blue
            label="open < close",
        )
    if "live_trades" in info:
        print(info["live_trades"])
        live_trades = [(unix_to_stamp(i["timestamp"]/1000), i["price"], i["side"]) for i in info["live_trades"]]
        buys = [(i[0], i[1]) for i in live_trades if i[2] == "buy"]
        sells = [(i[0], i[1]) for i in live_trades if i[2] == "sell"]
        if buys:
            axes[0].scatter(*zip(*buys), c="yellow", marker="^", s=120)
        if sells:
            axes[0].scatter(*zip(*sells), c="yellow", marker="v", s=120)

    # plot indicators
    plot_indicators(axes, timestamps, states, indicators, indicator_fmt)

    if len(states["trades"]) > 1:
        plot_trades(axes[0], states)

    for ax in axes[:-1]:
        ax.legend()
        ax.tick_params(axis="x", labelrotation=45)

    plot_balances(axes[-1], timestamps, states, data)

    plotmotion(block)
    return axes


def plot_trades(axis, states):
    # plot win / loss lines
    p_op = states["trades"][0]
    for op in states["trades"][1:]:
        color = "lime" if op.profit >= 1 else "tomato"
        axis.plot(
            unix_to_stamp([p_op.unix, op.unix]),
            [p_op.price, op.price],
            color=color,
            linewidth=2,
        )
        p_op = op

    # plot trade triangles
    buys = list(zip(
        *[
            [unix_to_stamp(op.unix), op.price]
            for op in states["trades"]
            if isinstance(op, Buy)
        ]
    ))
    sells = list(zip(
        *[
            [unix_to_stamp(op.unix), op.price]
            for op in states["trades"]
            if isinstance(op, Sell)
        ]
    ))
    overrides = list(zip(
        *[
            [unix_to_stamp(op.unix), op.price]
            for op in states["trades"]
            if op.is_override
        ]
    ))

    if overrides:
        axis.scatter(*overrides, c="yellow", marker="o", s=120)
    if buys:
        axis.scatter(*buys, c="lime", marker="^", s=80)
    if sells:
        axis.scatter(*sells, c="tomato", marker="v", s=80)


def plot_balances(axis, timestamps, states, data):
    # plot balances chart
    balances = rotate(states["balances"])

    (
        balances[data.asset],
        balances[data.currency],
    ) = compute_potential_balances(
        balances[data.asset],
        balances[data.currency],
        states["close"],
    )

    ax = None
    lines = []
    for idx, (token, balance) in list(enumerate(balances.items())):
        # handle parasite axes
        if ax is None:
            ax = axis
        else:
            ax = ax.twinx()
        # label the axis
        ax.set_ylabel(token)
        # make the line
        lines.append(
            ax.plot(
                timestamps,
                balance,
                label=token,
                color=["tomato", "yellow", "orange"][idx % 3],
            )[0]
        )
        # show only the y axis
        ax.tick_params(axis="y")
        ax.set_yscale("log")

    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)
    ax.set_title("Balances")


def compute_potential_balances(asset_balance, currency_balance, price):
    # Convert inputs to numpy arrays for efficient computation
    asset_balance = np.array(asset_balance)
    currency_balance = np.array(currency_balance)
    price = np.array(price)

    # Calculate the potential USD if all BTC were sold at current price
    potential_assets = currency_balance / price

    # Calculate the potential BTC if all USD were spent at current price
    potential_currency = asset_balance * price

    # Merge the actual BTC balance with the potential BTC balance
    merged_currency_balance = np.where(
        currency_balance > NIL, currency_balance, potential_currency
    )
    merged_asset_balance = np.where(
        asset_balance > NIL, asset_balance, potential_assets
    )

    return merged_asset_balance, merged_currency_balance
