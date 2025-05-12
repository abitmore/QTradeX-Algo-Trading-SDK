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
import os
import shutil
import time
from copy import deepcopy
from json import dumps as json_dumps
from multiprocessing import Manager, Process
from random import choice, choices, randint, random, sample
from statistics import median
from typing import Any, Dict, List

# 3RD PARTY MODULES
import matplotlib.pyplot as plt
import numpy as np
# QTRADEX MODULES
from qtradex.common.json_ipc import json_ipc
from qtradex.common.utilities import NonceSafe, it, print_table, sigfig
from qtradex.core import backtest
from qtradex.core.base_bot import Info
from qtradex.optimizers.qpso import QPSO, QPSOoptions
from qtradex.optimizers.utilities import (bound_neurons, end_optimization,
                                          merge, print_tune)
from qtradex.private.wallet import PaperWallet


class LSGAoptions(QPSOoptions):
    def __init__(self):
        super().__init__()
        self.population = 20
        self.offspring = 10
        self.top_ratio = 0.05
        self.processes = os.cpu_count() or 3
        self.fitness_ratios = None
        self.fitness_period = 20
        self.fitness_inversion = lambda x: dict(
            zip(x.keys(), [list(x.values())[-1]] + list(list(x.values())[:-1]))
        )
        self.cyclic_amplitude = 3
        self.cyclic_freq = 25
        self.erode = 0.9999
        self.erode_freq = 200
        self.temperature = 1  # 10 (coarse) to 0.0001 (fine)
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
    Print live updates and statistics during a session.
    """
    table = []
    table.append([""] + kwargs["parameters"] + [""] + kwargs["coords"] + [""])
    table.append(
        ["current test"]
        + [kwargs["bot"].tune[param] for param in kwargs["parameters"]]
        + [""]
        + [kwargs["new_score"][coord] for coord in kwargs["coords"]]
        + [""]
    )
    for coord, (score, bot) in kwargs["best_bots"].items():
        table.append(
            [coord]
            + [bot.tune[param] for param in kwargs["parameters"]]
            + [""]
            + [score[coord] for coord in kwargs["coords"]]
            + ["###"]
        )

    n_coords = len(kwargs["coords"])

    eye = np.eye(n_coords).astype(int)

    colors = np.vstack(
        (
            np.zeros((len(kwargs["parameters"]) + 2, n_coords + 2)),
            np.hstack(
                (
                    np.zeros((n_coords, 2)),
                    eye,
                )
            ),
            np.array(
                [
                    [0, 0]
                    + [
                        2 if i else 0
                        for i in kwargs["self"].options.fitness_ratios.values()
                    ]
                ]
            ),
        )
    )
    for coord in kwargs["boom"]:
        cdx = kwargs["coords"].index(coord)
        colors[len(kwargs["parameters"]) + 2 + cdx, cdx + 2] = 3

    msg = "\033c"
    msg += it(
        "green",
        f"Stochastic {len(kwargs['parameters'])}-Dimensional {n_coords} Coordinate "
        "Ascent with Pruned Neuroplasticity in Eroding Local Search Genetic Algorithm "
        "Optimization, Enhanced by Cyclic Simulated Annealing",
    )
    msg += "\n\n"
    msg += f"\n{print_table(table, render=True, colors=colors, pallete=[0, 34, 33, 178])}\n"
    msg += (
        f"\ntest {kwargs['iteration']} improvements {kwargs['improvements']} synapses"
        f" {len(kwargs['synapses'])}"
    )
    msg += f"\naegir {kwargs['aegir']}"
    msg += f"\n{kwargs['synapse_msg']} {it('yellow', kwargs['neurons'])}"
    msg += f"\n\n{((kwargs['n_backtests'] or 1)/(time.time()-kwargs['lsga_start'])):.2f} Backtests / Second"
    msg += f"\nRunning on {kwargs['self'].data.days} days of data."
    msg += "\n\nCtrl+C to quit and show final tune as copyable dictionary."
    print(msg)


def retest_process(bot, data, wallet, todo, done, **kwargs):
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
            done[work["id"]] = backtest(bot, data, wallet.copy(), plot=False, **kwargs)
    except KeyboardInterrupt:
        print("Compute process ending...")


class LSGA(QPSO):
    def __init__(self, data, wallet=None, options=None):
        if wallet is None:
            wallet = PaperWallet({data.asset: 0, data.currency: 1})
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

    def optimize(self, bot, **kwargs):
        """
        Perform Quantum Particle Swarm Optimization (QPSO) to optimize trading strategies.

        The function backtests and compares the results to previous best in terms of fitness (ROI).
        It intelligently chooses alternative parameters for the optimization process using various
        techniques such as N-Dimensional Brownian Drunk Walk, Dual Coordinate Gradient Ascent,
        Cyclic Simulated Annealing, Neuroplastic Synapse Connectivity, and Periodic Peak Fitness Digression.

        Parameters:
        bot (object): The trading bot to optimize.

        Returns:
        dict: The best bots and their associated performance metrics.
        """
        bot.info = Info({"mode": "optimize"})
        improvements = 0
        iteration = 0  # Tracks iterations during optimization
        idx = 0  # Index for fitness evaluation
        n_backtests = 1
        synapses = []  # Tracks past neuron connections

        # Reset the given bot and apply neuron boundaries
        bot.reset()
        # bot = bound_neurons(bot)

        # Initialize best_bots with initial backtest result
        initial_result = backtest(
            deepcopy(bot), self.data, deepcopy(self.wallet), plot=False, **kwargs
        )
        print("Initial Backtest:")
        print(json.dumps(initial_result, indent=4))

        coords = list(initial_result.keys())
        parameters = list(bot.tune.keys())

        best_bots = {coord: [initial_result.copy(), deepcopy(bot)] for coord in coords}

        # Initialize fitness ratios for all coordinates
        if self.options.fitness_ratios is None:
            self.options.fitness_ratios = {coord: 0 for coord in coords}
            self.options.fitness_ratios[coords[0]] = 1

        # Using multiprocessing to handle bot testing across processes
        with Manager() as manager:
            todo = manager.list()
            done = manager.dict()
            children = [
                Process(
                    target=retest_process,
                    args=(bot, self.data, self.wallet, todo, done),
                    kwargs=kwargs,
                )
                for _ in range(self.options.processes)
            ]
            for child in children:
                child.start()

            try:
                # Track start time for performance metrics
                lsga_start = time.time()

                while True:
                    # Optional pause to prevent CPU overload
                    time.sleep(self.options.cooldown)
                    iteration += 1
                    idx += 1

                    # Periodically adjust fitness ratios using user-defined function
                    if not idx % self.options.fitness_period:
                        self.options.fitness_ratios = self.options.fitness_inversion(
                            self.options.fitness_ratios
                        )

                    # Occasionally erode the best bot's scores to encourage alternative solutions
                    if idx % self.options.erode_freq == 0:
                        best_bots = {
                            coord: [
                                {k: v * self.options.erode for k, v in score.items()},
                                bot,
                            ]
                            for coord, (score, bot) in best_bots.items()
                        }

                    # Synaptogenesis: Considering new neuron connections
                    neurons = self.options.neurons or [
                        i for i in bot.tune.keys() if bot.clamps[i][3]
                    ]
                    for _ in range(3):
                        neurons = sample(population=neurons, k=randint(1, len(neurons)))
                    neurons.sort()

                    # Neuroplasticity: Select past winning synapses for adjustment
                    synapse_msg = ""
                    if randint(0, 2) and len(synapses) > 2:
                        synapse_msg = it("red", "synapse")
                        neurons = choice(synapses)

                    # Limit synapse count through pruning
                    synapses = list(set(synapses))[-self.options.synapses :]

                    # Select a random coordinate based on fitness ratio
                    coord = choices(
                        population=list(self.options.fitness_ratios.keys()),
                        weights=list(self.options.fitness_ratios.values()),
                        k=1,
                    )[0]
                    bot = best_bots[coord][1]

                    # Create population of bots for evaluation
                    bots = [deepcopy(bot) for _ in range(self.options.population)]
                    for botdx, bot in enumerate(bots):
                        # Quantum particle drunkwalk: Alter neurons in selected synapse
                        for neuron in neurons:
                            if not bot.clamps[neuron][3]:
                                continue

                            aegir, path = self.entheogen(
                                idx,
                                list(bot.tune.keys()).index(neuron) / len(bot.tune),
                                bot.tune[neuron].shape
                                if isinstance(bot.tune[neuron], np.ndarray)
                                else None,
                                bot.clamps[neuron][0],  # min
                                bot.clamps[neuron][2],  # max
                                # is it a numpy array of ints?
                                np.issubdtype(bot.tune[neuron].dtype, np.integer)
                                # if it is a numpy array,
                                if isinstance(bot.tune[neuron], np.ndarray)
                                # else is it a single int?
                                else isinstance(bot.tune[neuron], int),
                            )
                            bot.tune[neuron] += path

                        # Bound neurons to reasonable values
                        bound_neurons(bot)

                    new_scores = self.retest(todo, done, bots)

                    # Sort bots by fitness score for the selected coordinate
                    coordx = randint(0, len(self.options.fitness_ratios) - 1)
                    new_scores.sort(
                        key=lambda x: list(x[0].values())[coordx], reverse=True
                    )

                    # Select top performers
                    n_top = max(
                        int(self.options.population * self.options.top_ratio), 2
                    )
                    good_performers = sample(new_scores[:n_top], n_top)

                    # Merge best performers to create offspring
                    merged = [
                        merge(new_scores[0][1].tune, choice(good_performers)[1].tune)
                        for _ in range(self.options.offspring)
                    ]
                    bots = [deepcopy(bot) for _ in range(self.options.population)]
                    for bot, tune in zip(bots, merged):
                        bot.tune = tune

                    merged_scores = self.retest(todo, done, bots)

                    # Merge new scores with previous ones
                    new_scores.extend(merged_scores)
                    n_backtests += len(new_scores)

                    boom = []
                    improved = False
                    for new_score, bot in new_scores:
                        for coord, (check_score, check_bot) in best_bots.copy().items():
                            if new_score[coord] > check_score[coord]:
                                best_bots[coord] = (new_score, bot)
                                boom.append(coord)
                                improved = True

                    # Print relevant information and results if enabled
                    if self.options.show_terminal:
                        printouts(locals())

                    # If the bot improved, note the change and adjust iteration
                    if improved:
                        synapses.append(tuple(neurons))
                        iteration -= 1

                    # Check if optimization should stop
                    if (
                        idx > self.options.epochs
                        or iteration > self.options.improvements
                    ):
                        raise KeyboardInterrupt

            except KeyboardInterrupt:
                end_optimization(best_bots, self.options.print_tune)
                return best_bots
            finally:
                for child in children:
                    child.terminate()
