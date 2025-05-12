import importlib.util
import inspect
import os
import sys

import matplotlib.pyplot as plt
from qtradex.common.utilities import read_file
from qtradex.core.backtest import backtest


def reload_bot(filepath, bot_name):
    # Get the module name from the filepath
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    # Check if the module is already in sys.modules
    if module_name in sys.modules:
        # If it is, reload it
        module = importlib.reload(sys.modules[module_name])
    else:
        # If it is not, load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Add the module to sys.modules
        spec.loader.exec_module(module)

    # Get the class from the module
    return getattr(module, bot_name)


def auto_backtest(bot, data, wallet=None, **kwargs):
    # This function specifies these kwargs in backtest, so we can't have the user overriding them
    kwargs.pop("plot", None)
    kwargs.pop("block", None)

    # Create the wallet if it doesn't exist
    if wallet is None:
        wallet = PaperWallet({data.asset: 0, data.currency: 1})

    # What is the bot class called?
    botclass = bot.__class__.__name__
    # What is the contents of the botscript?
    botscript = inspect.getfile(type(bot))
    contents = read_file(botscript)
    # store the tune
    tune = bot.tune.copy()

    # Show an initial backtest
    print("\033c")
    backtest(bot, data, wallet, **kwargs, block=False, plot=True)
    while True:
        new_contents = read_file(botscript)
        # if the botscript changed
        # TODO: Ideally this is done with AST so that comments and whitespace don't trigger a plot
        if new_contents != contents:
            # then we need to update the plot
            bot = reload_bot(botscript, botclass)()
            bot.tune.update(tune)
            plt.clf()
            backtest(bot, data, wallet, **kwargs, block=False, plot=True)
            contents = new_contents

        plt.pause(1)
