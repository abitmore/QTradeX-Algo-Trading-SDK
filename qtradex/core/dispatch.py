import json
import os
import shutil
import sys
import time
from getpass import getpass
from random import choice, sample

import numpy as np
import qtradex as qx
from qtradex.common.utilities import it
from qtradex.core.tune_manager import choose_tune
from qtradex.core.tune_manager import load_tune as load_from_manager
from qtradex.core.ui_utilities import get_number, logo, select
from qtradex.private.wallet import PaperWallet


def load_tune(bot):
    options = [
        "Use best roi tune",
        "Use most recent best roi tune",
        "Use bot.tune",
        "Use bot.drop",
        "Use tune manager...",
    ]
    choice = select(options)
    if choice == 0:
        return load_from_manager(bot)
    elif choice == 1:
        return load_from_manager(bot, sort="latest")
    elif choice == 2:
        return bot.tune
    elif choice == 3:
        return {k: v[1] for k, v in bot.clamps.items()}
    elif choice == 4:
        return choose_tune(bot, "tune")

def plot_gravitas(bot, data, wallet, **kwargs):
    import matplotlib.pyplot as plt

    def get_float_input(prompt, default):
        user_input = input(f"{prompt} (default: {default}): ")
        return float(user_input) if user_input else default

    min_g = get_float_input("Min Gravitas", 0.3)
    max_g = get_float_input("Max Gravitas", 1.7)
    tests = int(get_float_input("Number of tests", 200.0))
    qx.backtest(bot, data, wallet.copy(), **kwargs, block=False)
    plt.figure("Gravitas")
    rois = []
    for g in np.linspace(min_g, max_g, tests):
        bot.gravitas = g
        rois.append(qx.backtest(bot, data, wallet.copy(), plot=False, **kwargs)["roi"])

    plt.plot(np.linspace(min_g, max_g, tests), rois)
    plt.ioff()
    plt.show()


def _flag(name, default=None):
    """Read --flag=value or --flag value from sys.argv."""
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    if name in sys.argv[1:]:
        idx = sys.argv[1:].index(name) + 1
        if idx < len(sys.argv[1:]):
            return sys.argv[1:][idx]
    return default

def _has(name):
    return any(a.startswith(name) for a in sys.argv[1:])

def _noninteractive_dispatch(bot, data, wallet, kwargs):
    args = sys.argv[1:]

    if _has("--help"):
        print("Usage: QTRADEX_NONINTERACTIVE=1 python botscript.py [flags]")
        print()
        print("Tune flags (default: best):")
        print("  --tune best      Use best ROI saved tune")
        print("  --tune latest    Use most recent saved tune")
        print("  --tune bot       Use bot.tune defaults")
        print("  --tune drop      Use midpoints from clamps")
        print()
        print("Action flags (default: backtest):")
        print("  --backtest       Run backtest")
        print("  --optimize [NAME] Run optimizer (QPSO, LSGA, IPSE, AION, GridSearch, RL)")
        print("  --papertrade     Run papertrade")
        print("  --autobacktest   Run auto backtest")
        print("  --monte-carlo    Run monte carlo simulation")
        print()
        print("Optimizer flags:")
        print("  --timeout SECONDS Stop optimizer after N seconds")
        return

    tune_flag = _flag("--tune", "best")

    resolve_tune = {
        "best": lambda b: load_from_manager(b),
        "latest": lambda b: load_from_manager(b, sort="latest"),
        "bot": lambda b: b.tune,
        "drop": lambda b: {k: v[1] for k, v in b.clamps.items()},
    }
    bot.tune = resolve_tune.get(tune_flag, resolve_tune["best"])(bot)

    if _has("--optimize"):
        opt_flag = _flag("--optimize", "QPSO")

        for k, v in bot.clamps.items():
            if len(v) == 2:
                bot.clamps[k] = (v[0], (v[0] + v[1]) / 2, v[1], 1)

        optimizers_map = {
            "QPSO": qx.optimizers.QPSO,
            "LSGA": qx.optimizers.LSGA,
            "IPSE": qx.optimizers.IPSE,
            "AION": qx.optimizers.AION,
            "GridSearch": qx.optimizers.GridSearch,
            "RL": qx.optimizers.RLPPO,
        }
        cls = optimizers_map[opt_flag]
        optimizer = cls(data, wallet)
        timeout = _flag("--timeout")
        if timeout:
            optimizer.options.timeout = float(timeout)
        optimizer.optimize(bot, **kwargs)
    elif _has("--papertrade"):
        qx.core.papertrade(bot, data, wallet, **kwargs)
    elif _has("--autobacktest"):
        qx.core.auto_backtest(bot, data, wallet, **kwargs)
    elif _has("--monte-carlo"):
        qx.core.monte_carlo(bot, data, wallet, **kwargs)
    else:
        qx.core.backtest(bot, data, wallet, plot=False, show=False, **kwargs)


def dispatch(bot, data, wallet=None, **kwargs):
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})

    if os.environ.get("QTRADEX_NONINTERACTIVE"):
        _noninteractive_dispatch(bot, data, wallet, kwargs)
        return

    logo(animate=True)

    bot.tune = load_tune(bot)
    options = [
        "Backtest",
        "Optimize",
        "Papertrade",
        "Live",
        "Show Fill Orders",
        "AutoBacktest",
        "Monte Carlo",
    ]
    choice = select(options)

    if choice == 0:
        qx.core.backtest(bot, data, wallet, **kwargs)
    elif choice == 1:
        for k, v in bot.clamps.items():
            if len(v) == 2:
                bot.clamps[k] = (v[0], (v[0]+v[1]) / 2, v[1], 1)

        options = [
            "QPSO (Quantum Particle Swarm Optimizer)",
            "LSGA (Local Search Genetic Algorithm)",
            "IPSE (Iterative Parametric Space Expansion)",
            "AION (Adaptive Intelligent Optimization Network)",
            "GridSearch",
            "Manual Tuner",
            "Gravitas",
        ]
        choice = select(options)

        if choice == 0:
            optimizer = qx.optimizers.QPSO(data, wallet)
        elif choice == 1:
            optimizer = qx.optimizers.LSGA(data, wallet)
        elif choice == 2:
            optimizer = qx.optimizers.IPSE(data, wallet)
        elif choice == 3:
            optimizer = qx.optimizers.AION(data, wallet)
        elif choice == 4:
            optimizer = qx.optimizers.GridSearch(data, wallet)
        elif choice == 5:
            optimizer = qx.optimizers.MouseWheelTuner(data, wallet)
        elif choice == 6:
            plot_gravitas(bot, data, wallet, **kwargs)
        if choice != 5:
            optimizer.optimize(bot, **kwargs)
    elif choice == 2:
        qx.core.papertrade(bot, data, wallet, **kwargs)
    elif choice in [3, 4]:
        if data.exchange == "bitshares":
            api_key = input("Enter username: ")
            api_secret = getpass("Enter WIF:      ")
        else:
            api_key = getpass("Enter API key:    ")
            api_secret = getpass("Enter API secret: ")

        if choice == 3:
            dust = input("Don't trade under this amount of assets (enter for 1e-8): ")
            if dust == "":
                dust = 1e-8
            else:
                dust = float(dust)

        if choice == 3:
            qx.core.live(bot, data, api_key, api_secret, dust, **kwargs)
        elif choice == 4:
            qx.core.filltest(bot, data, api_key, api_secret)
    elif choice == 5:
        qx.core.auto_backtest(bot, data, wallet, **kwargs)
    elif choice == 6:
        qx.core.monte_carlo(bot, data, wallet, **kwargs)
