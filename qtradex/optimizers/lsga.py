r"""
    .____       _________ ________    _____   
    |    |     /   _____//  _____/   /  _  \  
    |    |     \_____  \/   \  ___  /  /_\  \ 
    |    |___  /        \    \_\  \/    |    \
    |_______ \/_______  /\______  /\____|__  /
            \/        \/        \/         \/             
               
        ╦  ┌─┐┌─┐┌─┐┬    ╔═╗┌─┐┌─┐┬─┐┌─┐┬ ┬           
        ║  │ ││  ├─┤│    ╚═╗├┤ ├─┤├┬┘│  ├─┤           
        ╩═╝└─┘└─┘┴ ┴┴─┘  ╚═╝└─┘┴ ┴┴└─└─┘┴ ┴           
   ╔═╗┌─┐┌┐┌┌─┐┌┬┐┬┌─┐  ╔═╗┬  ┌─┐┌─┐┬─┐┬┌┬┐┬ ┬┌┬┐
   ║ ╦├┤ │││├┤  │ ││    ╠═╣│  │ ┬│ │├┬┘│ │ ├─┤│││
   ╚═╝└─┘┘└┘└─┘ ┴ ┴└─┘  ╩ ╩┴─┘└─┘└─┘┴└─┴ ┴ ┴ ┴┴ ┴


github.com/SquidKid-deluxe presents:

- N-Dimensional
- N Coordinate
- Stochastic Local Search Ascent
- with Pruned Neuroplasticity
- in an Eroding Genetic Algorithm Optimizer
- enhanced by Cyclic Simulated Annealing*

*really just a few if statements with a fancy picture frame
"""
# STANDARD MODULES
import getopt
import itertools
import json
import math
import shutil
import time
from copy import deepcopy
from json import dumps as json_dumps
from multiprocessing import Manager, Process
from random import choice, randint, random, sample
from statistics import median
from typing import Any, Dict, List

# 3RD PARTY MODULES
import matplotlib.pyplot as plt
import numpy as np
# QTRADEX MODULES
from qtradex.common.json_ipc import json_ipc
from qtradex.common.utilities import NonceSafe, it, print_table, sigfig
from qtradex.core import backtest
from qtradex.optimizers.qpso import QPSO, QPSOoptions
from qtradex.optimizers.utilities import (bound_neurons, end_optimization,
                                          merge, print_tune)


class LSGAoptions(QPSOoptions):
    def __init__(self):
        super().__init__()
        self.population = 20
        self.offspring = 10
        self.top_ratio = 0.05
        self.processes = 3
        self.fitness_ratios = (1, 0)
        self.fitness_period = 20
        self.fitness_inversion = lambda x: [x[-1]] + list(x[:-1])
        self.cyclic_amplitude = 3
        self.cyclic_freq = 25
        self.erode = 0.9999
        self.erode_freq = 200
        self.temperature = 3
        self.epochs = math.inf
        self.improvements = 10000
        self.cooldown = 0
        self.synapses = 50
        self.neurons = []
        self.show_terminal = True
        self.print_tune = False
        # path to write tunes to
        self.append_tune = ""


def printouts(kwargs):
    """
    Print live updates and statistics during a Quantum Particle Swarm Optimization (QPSO) session.
    """
    # Print statistics for solitary QPSO
    keys = list(kwargs["new_score"][1].tune.keys())

    table = []
    table.append([""] + keys + [""] + list(kwargs["new_score"][0].keys()) + [""])
    table.append(
        ["current test"]
        + list(kwargs["new_score"][1].tune.values())
        + [""]
        + list(kwargs["new_score"][0].values())
        + [""]
    )
    for n, (score, bot) in enumerate(kwargs["best_bots"]):
        table.append(
            [list(score.keys())[n]]
            + list(bot.tune.values())
            + [""]
            + list(score.values())
            + ["###"]
        )

    n_coords = len(kwargs["new_score"][0])

    eye = np.eye(n_coords).astype(int)

    colors = np.vstack(
        (
            np.zeros(
                (len(list(kwargs["new_score"][1].tune.values())) + 2, n_coords + 2)
            ),
            np.hstack(
                (
                    np.zeros((n_coords, 2)),
                    eye,
                )
            ),
            np.array(
                [[0, 0] + [int(i * 5) for i in kwargs["self"].options.fitness_ratios]]
            ),
        )
    )

    msg = "\033c"
    msg += it(
        "green",
        f"Stochastic {len(list(bot.tune.values()))}-Dimensional {n_coords} Coordinate "
        "Ascent with Pruned Neuroplasticity in Eroding Local Search Genetic Algorithm "
        "Optimization, Enhanced by Cyclic Simulated Annealing",
    )
    msg += "\n\n"
    msg += f"\n{print_table(table, render=True, colors=colors, pallete=[34, 34, 34, 34, 32, 33])}\n"
    msg += (
        f"\ntest {kwargs['iteration']} improvements {kwargs['improvements']} tweaks {kwargs['tweaks']} synapses"
        f" {len(kwargs['synapses'])}"
    )
    msg += f"\npath {kwargs['path']:.4f} aegir {kwargs['aegir']:.4f}"
    msg += f"\n{kwargs['synapse_msg']} {it('yellow', kwargs['neurons'])}"
    msg += f"\n\n{((kwargs['n_backtests'] or 1)/(time.time()-kwargs['lsga_start'])):.2f} Backtests / Second"
    msg += "\n\nCtrl+C to quit and show final tune as copyable dictionary."
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


class LSGA(QPSO):
    def __init__(self, data, wallet, options=None):
        self.options = options if options is not None else LSGAoptions()
        self.data = data
        self.wallet = wallet

    # check_improved and enthogen are inherited from QPSO

    def retest(self, todo, done, bots):
        # give jobs
        for bot_id, bot in enumerate(bots):
            todo.append({"id": bot_id, "tune": bot.tune})

        # wait for them to finish
        while len(done) < self.options.population:
            time.sleep(0.02)

        # gather results
        new_scores = []
        for bot_id, result in done.items():
            new_scores.append((result, bots[bot_id]))

        # clear the pipe
        done.clear()

        return new_scores

    def optimize(self, bot):
        """
        Perform Quantum Particle Swarm Optimization (QPSO) to optimize trading strategies.

        The function backtests and compares the results to previous best in terms of fitness (ROI).
        It intelligently chooses alternative parameters for the optimization process using various
        techniques such as N-Dimensional Brownian Drunk Walk, Dual Coordinate Gradient Ascent,
        Cyclic Simulated Annealing, Neuroplastic Synapse Connectivity, and Periodic Peak Fitness Digression.

        Parameters:
        storage (dict): The storage containing historical data and results.
        info (dict): Additional information required for the optimization process.
        data (list): The data used for backtesting.
        portfolio (dict): The portfolio details.
        mode (dict): The mode of the optimization process.
        tune (dict): The parameters for tuning the optimization.
        control (dict): The control parameters for the optimization.

        Returns:
        None
        """
        improvements = 0
        # NOTE: iteration is not monotonic, if the bot improves, "repeat condition"
        #       does iteration -= 1
        iteration = 0
        # idx is, though
        idx = 0
        n_backtests = 1
        synapses = []
        tweaks = 0
        dpt = 1

        # reset the given bot
        bot.reset()
        bot = bound_neurons(bot)

        # tuple of (score(s), bot)
        initial_results = (
            backtest(bot, self.data, self.wallet.copy(), plot=False),
            deepcopy(bot),
        )

        if len(initial_results[0]) != len(self.options.fitness_ratios):
            self.options.fitness_ratios = [1] + [0] * (len(initial_results[0]) - 1)

        # keep track of a best bot for each coordinate
        best_bots = [deepcopy(initial_results) for _ in self.options.fitness_ratios]

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
            # the whole function is wrapped in a try...except KeyboardInterrupt:
            # so that the tune is printed when the user presses Ctrl+C
            try:
                # note start time for speed tracking later
                lsga_start = time.time()
                while True:
                    # optional pause to keep cpu from pinning
                    time.sleep(self.options.cooldown)
                    # tick
                    iteration += 1
                    idx += 1

                    # Every `period`, use the user function to change coordinates
                    if not idx % self.options.fitness_period:
                        self.options.fitness_ratios = self.options.fitness_inversion(
                            self.options.fitness_ratios
                        )

                    # Occasionally erode the best bot's scores
                    # to allow for alternate n-dimensional travel
                    if idx % self.options.erode_freq == 0:
                        best_bots = [
                            ({k: v * self.options.erode for k, v in i[0].items()}, i[1])
                            for i in best_bots
                        ]

                    # Synaptogenesis considers a new neuron connection
                    if self.options.neurons:
                        # Allow an override
                        neurons = self.options.neurons
                    else:
                        neurons = list(bot.tune.keys())
                        for _ in range(3):
                            neurons = sample(
                                population=neurons, k=randint(1, len(neurons))
                            )
                    neurons.sort()

                    # Neuroplasticity considers past winning synapses
                    synapse_msg = ""
                    if randint(0, 2):
                        if len(synapses) > 2:
                            synapse_msg = it("red", "synapse")
                            neurons = choice(synapses)

                    # Synaptic pruning limits brain size
                    synapses = list(set(synapses))[-self.options.synapses :]

                    # create a population
                    bots = [deepcopy(bot) for _ in range(self.options.population)]

                    for bot in bots:
                        # Quantum particle drunkwalk alters neurons in the chosen synapse
                        for neuron in neurons:
                            aegir, path = self.entheogen(
                                idx,
                                list(bot.tune.keys()).index(neuron) / len(bot.tune),
                            )
                            # keep floats and integers
                            if isinstance(bot.tune[neuron], float):
                                bot.tune[neuron] *= path
                                bot.tune[neuron] += (random()-0.5)/1000
                            elif isinstance(bot.tune[neuron], int):
                                bot.tune[neuron] = bot.tune[neuron] + randint(
                                    -2, 2
                                )  # int(path * 5)
                        # Bound neurons to reasonable values
                        bot = bound_neurons(bot)

                    new_scores = self.retest(todo, done, bots)

                    # sort the bots, picking a random coordinate to use
                    coordx = randint(0, len(self.options.fitness_ratios) - 1)
                    new_scores.sort(
                        key=lambda x: list(x[0].values())[coordx], reverse=True
                    )

                    # pick some random good performers
                    n_top = max(
                        int(self.options.population * self.options.top_ratio), 2
                    )
                    good_performers = sample(new_scores[:n_top], n_top)

                    merged = [
                        merge(
                            new_scores[0][1].tune,
                            choice(good_performers)[1].tune,
                        )
                        for _ in range(self.options.offspring)
                    ]
                    bots = [deepcopy(bot) for _ in range(self.options.population)]
                    for bot, tune in zip(bots, merged):
                        bot.tune = tune

                    merged_scores = self.retest(todo, done, bots)

                    new_scores.extend(merged_scores)

                    n_backtests += len(new_scores)

                    for new_score in new_scores:
                        # Update variables and check if an improvement occurred
                        best_bots, improvements, boom, improved = self.check_improved(
                            new_score,
                            best_bots,
                            improvements,
                        )

                    # Print relevant information and results
                    if self.options.show_terminal:
                        printouts(locals())

                    # if the bot got better
                    if improved:
                        # note the synapses
                        synapses.append(tuple(neurons))
                        # keep it for the next round
                        bot = new_score[1]
                        # repeact condition
                        iteration -= 1

                    # if we're done
                    if (
                        idx > self.options.epochs
                        or iteration > self.options.improvements
                    ):
                        # easiest way to not duplicate code;
                        # the try..except handles tune export
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                end_optimization(best_bots, self.options.print_tune)
                return best_bots
            finally:
                for child in children:
                    child.terminate()
