import json
import shutil
import time
from getpass import getpass
from random import choice, sample

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


def dispatch(bot, data, wallet=None):
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})
    logo(animate=True)

    bot.tune = load_tune(bot)
    options = [
        "Backtest",
        "Optimize",
        "Papertrade",
        "Live",
    ]
    choice = select(options)

    if choice == 0:
        qx.core.backtest(bot, data, wallet)
    elif choice == 1:
        options = [
            "QPSO (Quantum Particle Swarm Optimizer)",
            "LSGA (Local Search Genetic Algorithm)",
            "IPSE (Iterative Parametric Space Expansion)",
            "Manual Tuner"
        ]
        choice = select(options)

        if choice == 0:
            optimizer = qx.optimizers.QPSO(data, wallet)
        elif choice == 1:
            optimizer = qx.optimizers.LSGA(data, wallet)
        elif choice == 2:
            optimizer = qx.optimizers.IPSE(data, wallet)
        elif choice == 3:
            optimizer = qx.optimizers.MouseWheelTuner(data, wallet)
        optimizer.optimize(bot)
    elif choice == 2:
        qx.core.papertrade(bot, data, wallet)
    elif choice == 3:
        if data.exchange == "bitshares":
            api_key = input("Enter username: ")
            api_secret = getpass("Enter WIF:      ")
        else:
            api_key = getpass("Enter API key:    ")
            api_secret = getpass("Enter API secret: ")

        dust = input("Don't trade under this amount of assets (enter for 1e-8): ")
        if dust == "":
            dust = 1e-8
        else:
            dust = float(dust)

        # TODO:
        # some kind of login menu, currently an error is thrown if the key isn't valid
        qx.core.live(bot, data, api_key, api_secret, dust)
