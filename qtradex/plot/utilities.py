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


def plot_indicators(axes, states, indicators, indicator_fmt):
    for key, name, color, idx, title in indicator_fmt:
        ax = axes[idx]
        ax.set_title(title)
        # Plot each EMA with a color gradient
        ax.plot(
            states["unix"],
            indicators[key],
            color=color,
            label=name,
        )


def plot(data, states, indicators, block, indicator_fmt, style="dark_background"):
    mplstyle.use(style)
    # DONE plotting of buy/sell with win/loss line plotting
    # DONE buy/sell are green/red triangles
    # DONE plotting of high/low/open/close
    # DONE plotting of indicators (dict of indicator keys to be plotted and color)
    # YIKES balance plotting follows price on token not held

    n_levels = max(i[3] for i in indicator_fmt) + 2
    # clear the current figure
    plt.clf()
    axes = [plt.subplot(n_levels, 1, n) for n in range(1, n_levels + 1)]
    axes[0].set_yscale("log")

    # plotting of high/low/open/close
    # high/low
    axes[0].fill_between(
        states["unix"],
        states["low"],
        states["high"],
        color="magenta",
        alpha=0.3,
        label="High/Low",
    )
    # Fill between for open > close
    axes[0].fill_between(
        states["unix"],
        states["open"],
        states["close"],
        where=expand_bools(states["open"] > states["close"], side="right"),
        color=(1, 0, 0, 0.3),  # Red for open > close
        label="Open > Close",
    )

    # Fill between for open < close
    axes[0].fill_between(
        states["unix"],
        states["open"],
        states["close"],
        where=expand_bools(states["open"] < states["close"], side="right"),
        color=(0, 1, 0, 0.3),  # Green for open < close
        label="Open < Close",
    )

    # plot indicators
    plot_indicators(axes, states, indicators, indicator_fmt)

    if len(states["trades"]) > 1:
        # plot win / loss lines
        p_op = states["trades"][0]
        for op in states["trades"][1:]:
            color = "lime" if op.profit >= 1 else "tomato"
            axes[0].plot(
                [p_op.unix, op.unix], [p_op.price, op.price], color=color, linewidth=2
            )
            p_op = op

        # plot trade triangles
        buys = zip(
            *[[op.unix, op.price] for op in states["trades"] if isinstance(op, Buy)]
        )
        sells = zip(
            *[[op.unix, op.price] for op in states["trades"] if isinstance(op, Sell)]
        )

        axes[0].scatter(*buys, c="lime", marker="^", s=80)
        axes[0].scatter(*sells, c="tomato", marker="v", s=80)

    for ax in axes[:-1]:
        ax.legend()

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
            ax = axes[n_levels - 1]
        else:
            ax = ax.twinx()
        # label the axis
        ax.set_ylabel(token)
        # make the line
        lines.append(
            ax.plot(
                states["unix"],
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

    plotmotion(block)
    return axes


def compute_potential_balances(asset_balance, currency_balance, price):
    # plt.clf()
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

    # plt.subplot(211)
    # plt.plot(asset_balance, linestyle="-", label="orig")
    # plt.plot(potential_assets, linestyle="--", label="potential")
    # plt.plot(merged_asset_balance, linestyle=":", label="merged")
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(currency_balance, linestyle="-", label="orig")
    # plt.plot(potential_currency, linestyle="--", label="potential")
    # plt.plot(merged_currency_balance, linestyle=":", label="merged")
    # plt.legend()
    # plt.show()
    return merged_asset_balance, merged_currency_balance
