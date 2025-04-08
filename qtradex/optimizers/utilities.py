import json
import math
import random

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import qtradex as qx

mplstyle.use("dark_background")
plt.rcParams["figure.raise_window"] = False


def bound_neurons(bot):
    def clamp(value, minv, maxv, strength):
        """
        clamp `value` between `minv` and `maxv` with `strength`
        if strength is one, value is hard clipped
        if strength is 0.5, value is returned as the mean of itself and any boundries it is outside of
        if strength is 0, it is returned as is

        this works for all values of strength between 0 and 1
        """

        isint = isinstance(value, int)

        ret = None
        # if told not to tune or value is within range
        if not strength or minv <= value <= maxv:
            # don't touch
            ret = value
        # less than minimum
        elif value < minv:
            ret = (value * (1 - strength)) + (minv * strength)
        # more than maximum
        elif value > maxv:
            ret = (value * (1 - strength)) + (maxv * strength)
        return int(ret) if isint else ret

    bot.tune = {
        key: clamp(bot.tune[key], minv, maxv, clamp_amt)
        for key, (minv, _, maxv, clamp_amt) in bot.clamps.items()
    }

    bot.autorange()
    return bot


def print_tune(score, bot, render=False):
    msg = ""
    just = max(map(len, score))
    for k, s in score.items():
        msg += f"# {k}".ljust(just + 2) + f" {s:.3f}\n"

    msg += "self.tune = " + json.dumps(bot.tune, indent=4)
    msg += "\n\n"
    if not render:
        print(msg)
    return msg


def end_optimization(best_bots, show):
    msg = "\033c=== FINAL TUNES ===\n\n"
    for n, (score, bot) in enumerate(best_bots):
        name = f"BEST {list(score.keys())[n].upper()} TUNE"
        msg += "## " + name + "\n\n"
        msg += print_tune(score, bot, render=True)
        bot.tune = {"tune":bot.tune, "results":score}
        qx.core.tune_manager.save_tune(bot, name)
    if show:
        print(msg)


def merge(tune1, tune2):
    tune3 = {}
    for k, v in tune1.items():
        value = (random.random() / 2) + 0.25
        if isinstance(v, int):
            tune3[k] = int(round((v * value) + (tune2[k] * (1 - value))))
        else:
            tune3[k] = (v * value) + (tune2[k] * (1 - value))
    return tune3


def plot_scores(historical, historical_tests, cdx):
    """
    historical is a matrix like this:
    [
        (
            idx,
            [
                (score, bot),
                (score, bot),
                ...
            ]
        )
    ]
    """
    if not historical:
        return
    plt.clf()
    n_coords = len(historical[0][1])
    # initialize empty lists
    lines = [[] for _ in range(n_coords)]
    x_list = []
    for mdx, moment in enumerate(historical):
        x_list.append(moment[0])
        if mdx:
            x_list.append(moment[0])
            for idx in range(n_coords):
                lines[idx].append(lines[idx][-1])

        for idx, (score, _) in enumerate(moment[1]):
            lines[idx].append(list(score.values())[idx])
    x_list.append(cdx)
    for idx in range(n_coords):
        lines[idx].append(lines[idx][-1])

    sqrt = n_coords**0.5

    height = math.ceil(sqrt)
    width = height

    x_list_tests = [i[0] for i in historical_tests]

    for idx in range(n_coords):
        plt.subplot(width, height, idx + 1)
        plt.title(list(historical[0][1][0][0].keys())[idx])
        plt.plot(x_list, lines[idx], color="green")
        plt.xscale("log")
        plt.scatter(
            x_list_tests,
            [i[1][idx] for i in historical_tests],
            color="yellow",
        )
    plt.tight_layout()
    plt.pause(0.1)
