r"""
 ________ __________  _________________   
  \_____  \\______   \/   _____/\_____  \  
   /  / \  \|     ___/\_____  \  /   |   \ 
  /   \_/.  \    |    /        \/    |    \
  \_____\ \_/____|   /_______  /\_______  /
         \__>                \/         \/                       
               
╔═╗ ┬ ┬┌─┐┌┐┌┌┬┐┬ ┬┌┬┐  ╔═╗┌─┐┬─┐┌┬┐┬┌─┐┬  ┌─┐
║═╬╗│ │├─┤│││ │ │ ││││  ╠═╝├─┤├┬┘ │ ││  │  ├┤ 
╚═╝╚└─┘┴ ┴┘└┘ ┴ └─┘┴ ┴  ╩  ┴ ┴┴└─ ┴ ┴└─┘┴─┘└─┘
  ╔═╗┬ ┬┌─┐┬─┐┌┬┐  ╔═╗┌─┐┌┬┐┬┌┬┐┬┌─┐┌─┐┬─┐    
  ╚═╗│││├─┤├┬┘│││  ║ ║├─┘ │ │││││┌─┘├┤ ├┬┘    
  ╚═╝└┴┘┴ ┴┴└─┴ ┴  ╚═╝┴   ┴ ┴┴ ┴┴└─┘└─┘┴└─    


github.com/litepresence & github.com/SquidKid-deluxe present:

Elitist 
Spiral Bred 
Stochastic Dual Coordinate 
Eroding Ascent
Pruned Neuroplastic 
Quantum Particle Swarm Optimization
with Cyclic Simulated Annealing*

*really just a few if statements with a fancy picture frame
"""

# FIXME
# plotting for qpso yellow dots/green max
# plotting for qpso cyclic annealing
# plotting for qpos backtest


# STANDARD MODULES
import getopt
import itertools
import json
import math
import shutil
import time
from copy import deepcopy
from json import dumps as json_dumps
from multiprocessing import Process
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
from qtradex.optimizers.utilities import (bound_neurons, end_optimization,
                                          plot_scores, print_tune)

NIL = 10 / 10**10


class QPSOoptions:
    def __init__(self):
        self.lag = 0.5
        # plot this top percent of candidates
        self.top_percent = 0.9
        self.plot_period = 100
        self.fitness_ratios = (1, 0)
        self.fitness_period = 200
        self.fitness_inversion = lambda x: [x[-1]] + list(x[:-1])
        self.cyclic_amplitude = 3
        self.cyclic_freq = 1000
        self.digress = 0.99
        self.digress_freq = 2500
        self.temperature = 2
        self.epochs = math.inf
        self.improvements = 100000
        self.cooldown = 0
        self.synapses = 50
        self.neurons = []
        self.show_terminal = True
        self.print_tune = False
        # path to write tunes to
        self.append_tune = ""


def value_to_xterm_color(value):
    if not (0 <= value <= 1):
        raise ValueError("Value must be between 0 and 1.")

    # Calculate the red and green components
    red = int(196 * (1 - value))  # 196 is the xterm color code for red
    green = int(46 * value)  # 46 is the xterm color code for green

    # Combine the colors into a single xterm color code
    # The formula for combining is: 16 + (red * 36) + (green * 6)
    color_code = 16 + (red * 36) + (green * 6)

    return color_code


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
        "Ascent with Pruned Neuroplasticity in Eroding Quantum Particle Swarm "
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
    msg += f"\n\n{((kwargs['idx'] or 1)/(time.time()-kwargs['qpso_start'])):.2f} Backtests / Second"
    msg += "\n\nCtrl+C to quit and show final tune as copyable dictionary."
    print(msg)


class QPSO:
    def __init__(self, data, wallet, options=None):
        self.options = options if options is not None else QPSOoptions()
        self.data = data
        self.wallet = wallet

    def check_improved(self, new_score, best_bots, improvements):
        """
        Check for improvement upon dual gradient ascent and note the improvement.

        Parameters:
        - improvements (int): Number of improvements made during the session.

        Returns:
        - tuple: Tuple containing updated backup storage, storage, number of improvements,
        boom message, and improvement status.
        """

        improved = False
        boom = ""

        value = random()

        # Stochastic N-Coordinate Gradient Ascent
        for coordinate, (score, _) in enumerate(best_bots):
            key = list(score.keys())[coordinate]
            if new_score[0][key] > score[key]:
                if value < self.options.fitness_ratios[coordinate]:
                    improved = True
                boom += f"!!! BOOM {key.upper()} !!!\n"
                old = best_bots[coordinate][0].copy()

                best_bots[coordinate] = deepcopy(new_score)
                best_bots[coordinate][0][key] = (old[key] * self.options.lag) + (
                    best_bots[coordinate][0][key] * (1 - self.options.lag)
                )

        if improved:
            improvements += 1

        return best_bots, improvements, it("green", boom), improved

    def entheogen(self, i, j):
        """
        Somnambulism induces altered states of consciousness.
            The update equation is designed to guide the particles towards
            the optimal solution within the search space.
            Just like sleepwalking, where individuals exhibit automatic and
            seemingly purposeful actions while asleep, the movement of particles
            in QPSO can also appear to be random or unconscious, yet guided
            by certain optimization principles.
        Args:
            i (int): The iteration number.
        Returns:
            Tuple[float, float]: A tuple containing the calculated `aegir` and `path` values.
        """
        # Cyclic simulated annealing
        aegir = (
            self.options.cyclic_amplitude
            * math.sin(((i / self.options.cyclic_freq) + j) * math.pi * 2)
            + 1
            + self.options.cyclic_amplitude
        )
        # Quantum orbital pulse
        path = 1 + random() * (self.options.temperature * aegir) ** (1 / 2.0) / 100.0
        # Brownian iterator
        if randint(0, 1) == 0:
            path = 1 / path
        # Stumbling, yet remarkably determined
        return aegir, path

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

        # keep track of the past best bots
        historical = []
        # and those that were "kinda good"
        historical_tests = []

        if self.options.plot_period:
            plt.ion()

        # the whole function is wrapped in a try...except KeyboardInterrupt:
        # so that the tune is printed when the user presses Ctrl+C
        try:
            # note start time for speed tracking later
            qpso_start = time.time()
            while True:
                # optional pause to keep cpu from pinning
                time.sleep(self.options.cooldown)
                # tick
                iteration += 1
                idx += 1
                if self.options.plot_period and not idx % self.options.plot_period:
                    plot_scores(historical, historical_tests, idx)

                # Every `period`, use the user function to change coordinates
                if not idx % self.options.fitness_period:
                    self.options.fitness_ratios = self.options.fitness_inversion(
                        self.options.fitness_ratios
                    )

                # Occasionally erode the best bot's scores
                # to allow for alternate n-dimensional travel
                if iteration % self.options.digress_freq == 0:
                    best_bots = [
                        ({k: v * self.options.digress for k, v in i[0].items()}, i[1])
                        for i in best_bots
                    ]

                # Synaptogenesis considers a new neuron connection
                if self.options.neurons:
                    neurons = self.options.neurons
                else:
                    neurons = list(bot.tune.keys())
                    for _ in range(3):
                        neurons = sample(population=neurons, k=randint(1, len(neurons)))
                neurons.sort()

                # Neuroplasticity considers past winning synapses
                synapse_msg = ""
                if randint(0, 2):
                    if len(synapses) > 2:
                        synapse_msg = it("red", "synapse")
                        neurons = choice(synapses)

                # Synaptic pruning limits brain size
                synapses = list(set(synapses))[-self.options.synapses :] if self.options.synapses else []

                # Quantum particle drunkwalk alters neurons in the chosen synapse
                for neuron in neurons:
                    aegir, path = self.entheogen(
                        iteration, list(bot.tune.keys()).index(neuron) / len(bot.tune)
                    )
                    # keep floats and integers
                    if isinstance(bot.tune[neuron], float):
                        bot.tune[neuron] *= path
                        bot.tune[neuron] += (random() - 0.5) / 1000
                    elif isinstance(bot.tune[neuron], int):
                        bot.tune[neuron] = int(
                            bot.tune[neuron] + randint(-2, 2)
                        )  # + int(path * 5)

                # Bound neurons to reasonable values
                bot = bound_neurons(bot)

                # Perform a new backtest
                new_score = (
                    backtest(bot, self.data, self.wallet.copy(), plot=False),
                    bot,
                )

                # Update variables and check if an improvement occurred
                best_bots, improvements, boom, improved = self.check_improved(
                    new_score,
                    best_bots,
                    improvements,
                )

                # Print relevant information and results
                if self.options.show_terminal and not idx % 10:
                    printouts(locals())

                # if the bot got better
                if improved:
                    # note the synapses
                    synapses.append(tuple(neurons))
                    # keep it for the next round
                    bot = new_score[1]
                    # this is a "historical" moment, so keep the new best bots
                    historical.append((idx, deepcopy(best_bots)))
                    # repeact condition
                    iteration -= 1

                # if this bot is decent in any regard, save its score for plotting
                for coord, (score, _) in enumerate(best_bots):
                    nv = list(new_score[0].values())
                    if (
                        nv[coord]
                        >= list(score.values())[coord] * self.options.top_percent
                    ):
                        historical_tests.append((idx, nv))
                        break

                # if we're done
                if idx > self.options.epochs or iteration > self.options.improvements:
                    # easiest way to not duplicate code;
                    # the try..except handles tune export
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            end_optimization(best_bots, self.options.print_tune)
            return best_bots
