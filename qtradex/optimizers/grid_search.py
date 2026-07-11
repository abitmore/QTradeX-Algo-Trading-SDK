import math
import os
import time
from copy import deepcopy
from itertools import product
from multiprocessing import Manager, Process
from random import sample


class GridSearchOptions:
    def __init__(self):
        self.iterations = 50
        self.grid_dims = 2
        self.grid_points = 10
        self.processes = os.cpu_count() or 4
        self.grid_margin = 0.0  # fraction to trim from each clamp edge (0.15 = 30% interior only)
        self.show_terminal = True
        self.print_tune = False
        self.epochs = math.inf
        self.improvements = math.inf
        self.select_data = False


def _worker(data, wallet, todo, done, kwargs):
    from qtradex.core import backtest

    try:
        while True:
            try:
                work = todo.pop(0)
            except IndexError:
                time.sleep(0.02)
                continue
            result = backtest(
                deepcopy(work["bot"]), data, deepcopy(wallet), plot=False, **kwargs
            )
            done[work["id"]] = result
    except KeyboardInterrupt:
        pass


class GridSearch:
    def __init__(self, data, wallet=None, options=None):
        if wallet is None:
            raise ValueError("Wallet required")
        self.options = options or GridSearchOptions()
        self.data = data
        self.wallet = wallet

    def optimize(self, bot, **kwargs):
        from qtradex.core import backtest
        from qtradex.core.base_bot import Info

        bot.info = Info({"mode": "optimize"})
        bot.reset()

        initial_result = backtest(
            deepcopy(bot), self.data, deepcopy(self.wallet), plot=False, **kwargs
        )

        best_bots = {"sortino_ratio": [initial_result.copy(), deepcopy(bot)]}

        params = [p for p in bot.tune.keys() if bot.clamps[p][3]]
        if not params:
            if self.options.print_tune:
                from qtradex.optimizers.utilities import print_tune as pt
                pt(best_bots["sortino_ratio"][0], best_bots["sortino_ratio"][1])
            return best_bots

        with Manager() as manager:
            todo = manager.list()
            done = manager.dict()
            children = [
                Process(
                    target=_worker,
                    args=(self.data, self.wallet, todo, done, kwargs),
                )
                for _ in range(self.options.processes)
            ]
            for c in children:
                c.start()

            try:
                for iteration in range(self.options.iterations):
                    if len(params) < self.options.grid_dims:
                        grid_params = params
                    else:
                        grid_params = sample(
                            params, min(self.options.grid_dims, len(params))
                        )

                    current_best = deepcopy(best_bots["sortino_ratio"][1])

                    param_values = []
                    for p in grid_params:
                        lo, mid, hi, flag = bot.clamps[p]
                        span = hi - lo
                        margin = span * self.options.grid_margin
                        lo_trim = lo + margin
                        hi_trim = hi - margin
                        step = (hi_trim - lo_trim) / (self.options.grid_points - 1) if self.options.grid_points > 1 else 0
                        values = [lo_trim + i * step for i in range(self.options.grid_points)]
                        param_values.append(values)

                    n_points = 1
                    for v in param_values:
                        n_points *= len(v)

                    grid_values = []
                    for combo in product(*param_values):
                        entry = {}
                        for p, val in zip(grid_params, combo):
                            entry[p] = val
                        grid_values.append(entry)

                    for i in range(n_points):
                        candidate = deepcopy(current_best)
                        for p, val in grid_values[i].items():
                            candidate.tune[p] = val
                        todo.append({"id": i, "bot": candidate})

                    while len(done) < n_points:
                        time.sleep(0.02)

                    for i in range(n_points):
                        result = done.pop(i, None)
                        if result is None:
                            continue
                        sortino = result.get("sortino_ratio", -999)
                        current_best_sortino = best_bots["sortino_ratio"][0].get(
                            "sortino_ratio", -999
                        )
                        if sortino > current_best_sortino:
                            candidate = deepcopy(current_best)
                            for p, val in grid_values[i].items():
                                candidate.tune[p] = val
                            best_bots["sortino_ratio"] = [result, candidate]

                    if self.options.show_terminal:
                        best = best_bots["sortino_ratio"]
                        print(
                            f"Iter {iteration+1}/{self.options.iterations} | "
                            f"Grid: {grid_params} | "
                            f"Best sortino: {best[0].get('sortino_ratio', 'N/A'):.4f} | "
                            f"ROI: {best[0].get('roi', 'N/A'):.4f}"
                        )

            except KeyboardInterrupt:
                pass
            finally:
                for c in children:
                    c.terminate()

        if self.options.print_tune:
            from qtradex.optimizers.utilities import print_tune as pt
            pt(best_bots["sortino_ratio"][0], best_bots["sortino_ratio"][1])

        return best_bots
