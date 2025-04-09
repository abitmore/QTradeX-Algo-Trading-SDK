r"""
            ._____________  ____________________
            |   \______   \/   _____/\_   _____/
            |   ||     ___/\_____  \  |    __)_ 
            |   ||    |    /        \ |        \
            |___||____|   /_______  //_______  /
                                  \/         \/ 

    ╦┌┬┐┌─┐┬─┐┌─┐┌┬┐┬┬  ┬┌─┐  ╔═╗┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐┬─┐┬┌─┐
    ║ │ ├┤ ├┬┘├─┤ │ │└┐┌┘├┤   ╠═╝├─┤├┬┘├─┤│││├┤  │ ├┬┘││  
    ╩ ┴ └─┘┴└─┴ ┴ ┴ ┴ └┘ └─┘  ╩  ┴ ┴┴└─┴ ┴┴ ┴└─┘ ┴ ┴└─┴└─┘
          ╔═╗┌─┐┌─┐┌─┐┌─┐  ╔═╗─┐ ┬┌─┐┌─┐┌┐┌┌─┐┬┌─┐┌┐┌     
          ╚═╗├─┘├─┤│  ├┤   ║╣ ┌┴┬┘├─┘├─┤│││└─┐││ ││││     
          ╚═╝┴  ┴ ┴└─┘└─┘  ╚═╝┴ └─┴  ┴ ┴┘└┘└─┘┴└─┘┘└┘     

"""

# STANDARD MODULES
import json
import math
import time
from copy import deepcopy
from multiprocessing import Manager, Process

# 3RD PARTY MODULES
import numpy as np
# QTRADEX MODULES
from qtradex.common.utilities import NonceSafe, it, print_table, sigfig
from qtradex.core import backtest
from qtradex.optimizers.utilities import (bound_neurons, end_optimization,
                                          plot_scores, print_tune)
from qtradex.private.wallet import PaperWallet

NIL = 10 / 10**10


class IPSEoptions:
    def __init__(self):
        self.acceleration = 0.8
        self.space_size = 25
        self.processes = 3
        self.show_terminal = True
        self.print_tune = False


def printouts(kwargs):
    """
    Print live updates and statistics.
    """
    # Print statistics for solitary IPSE
    params = list(kwargs["bot"].tune.keys())
    coords = list(kwargs["score"].keys())

    table = []
    table.append([""] + params + [""] + coords)

    for n, (score, bot) in enumerate(kwargs["best_bots"].values()):
        table.append(
            [list(score.keys())[n]]
            + list(bot.tune.values())
            + [""]
            + list(score.values())
        )

    n_coords = len(kwargs["score"])

    eye = np.eye(n_coords).astype(int)

    colors = np.vstack(
        (
            np.zeros((len(list(kwargs["bot"].tune.values())) + 2, n_coords + 1)),
            np.hstack(
                (
                    np.zeros((n_coords, 1)),
                    eye,
                )
            ),
        )
    )
    msg = "\033c\n"
    colors[params.index(kwargs["parameter"]) + 1][
        coords.index(kwargs["coordinate"]) + 1
    ] = 2
    for i in kwargs["improved"]:
        colors[params.index(i[1]) + 1][coords.index(i[0]) + 1] = 3

    msg += (
        f"{print_table(table, render=True, colors=colors, pallete=[15, 34, 33, 178])}\n"
    )
    msg += f"\nexpansions - {kwargs['expansions']}"
    msg += f"\nepoch {kwargs['epoch']} - Optimized '{kwargs['parameter']}' by {kwargs['coordinate']}"
    msg += f"\n\n{((kwargs['idx'] or 1)/(time.time()-kwargs['ipse_start'])):.2f} Backtests / Second"
    msg += "\n\nCtrl+C to quit and save tune."
    print(msg)


def retest_process(bot, data, wallet, todo, done):
    try:
        while True:
            try:
                # get the work
                work = todo.pop(0)
            except IndexError:
                # wait for work
                time.sleep(0.02)
                continue
            # assign the tune
            bot.tune = work["tune"]
            # backtest and put in the done dictionary
            done[work["id"]] = backtest(bot, data, wallet.copy(), plot=False)
    except KeyboardInterrupt:
        print("Compute process ending...")

class IPSE:
    def __init__(self, data, wallet=None, options=None):
        if wallet is None:
            wallet = PaperWallet({data.asset: 0, data.currency: 1})
        self.options = options if options is not None else IPSEoptions()
        self.data = data
        self.wallet = wallet

    def retest(self, todo, done, bot, space, parameter):
        # give jobs
        for bot_id, test in enumerate(space):
            todo.append({"id": bot_id, "tune": {**bot.tune, parameter: test}})

        # wait for them to finish
        while len(done) < len(space):
            time.sleep(0.02)

        # gather results
        new_scores = []
        for bot_id, result in done.items():
            new_scores.append((bot_id, result))

        # clear the pipe
        done.clear()

        return [i[1] for i in sorted(new_scores, key=lambda x:x[0])]

    def optimize(self, bot):
        bot.reset()
        bot = bound_neurons(bot)
        coords = backtest(deepcopy(bot), self.data, self.wallet.copy(), plot=False)
        print("Initial Backtest:")
        print(json.dumps(coords, indent=4))

        ranges = {parameter: [i[0], i[2]] for parameter, i in bot.clamps.items()}
        score = coords.copy()

        best_bots = {coord: [score.copy(), deepcopy(bot)] for coord in coords}

        ipse_start = time.time()
        idx = 0
        epoch = 0
        expansions = 0
        # multiprocessing manager for shared ipc dictionary
        with Manager() as manager:
            todo = manager.list()
            done = manager.dict()
            children = [
                Process(
                    target=retest_process,
                    args=(bot, self.data, self.wallet, todo, done),
                )
                for _ in range(self.options.processes)
            ]
            for child in children:
                child.start()
            try:
                while True:
                    epoch += 1
                    # for each coordinate
                    for coordinate in coords:
                        # for each parameter
                        for parameter in bot.tune:
                            # Create a space of n evenly spaced points plus the one we were given
                            space = np.linspace(
                                *ranges[parameter], self.options.space_size
                            ).tolist() + [bot.tune[parameter]]

                            scores = self.retest(todo, done, bot, space, parameter)
                            idx += len(space)

                            improved = []
                            for check_coord in coords:
                                # select the best value of the current corrdinate
                                best_idx = np.argmax([i[check_coord] for i in scores])
                                score = scores[best_idx]
                                best = space[best_idx]

                                if (
                                    best_bots[check_coord][0][check_coord]
                                    < score[check_coord]
                                ):
                                    improved.append([check_coord, parameter])
                                    # assign this best value to the bot's tune
                                    best_bots[check_coord][1].tune[parameter] = best
                                    best_bots[check_coord][0] = score

                            # show optimization statistics
                            if self.options.show_terminal:
                                printouts(locals())
                    # Space Expansion
                    for parameter, value in bot.tune.items():
                        # zoom into the space near this "best" point
                        # and do a finer search next time.  Accomplished by taking the
                        # weighted average of the min/max points and the best point
                        ranges[parameter][0] = (
                            ranges[parameter][0] * self.options.acceleration
                        ) + (value * (1 - self.options.acceleration))
                        ranges[parameter][1] = (
                            ranges[parameter][1] * self.options.acceleration
                        ) + (value * (1 - self.options.acceleration))
                    expansions += 1
            except KeyboardInterrupt:
                end_optimization(list(best_bots.values()), self.options.print_tune)
                return best_bots
