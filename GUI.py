#  starts with the default parameters used in the report
"""
GUI.py - interactive view of the model, built on the course's pyics framework.

Every default comes from config.json, so the GUI starts with the parameters
used in the report (an assignment requirement). Changing a value in the GUI
and pressing Reset reruns the model with it.

Two modes (switch with the show_ga parameter):
    show_ga = False   replicator dynamics of the fixed field; one Step = one generation
    show_ga = True    genetic algorithm against the fixed field; one Step = one GA generation

Usage:  python GUI.py
"""

import matplotlib.pyplot as plt
from pyics import Model

import Strategies
import model
from GA import GeneticAlgorithm
from experiments import load_config, strategy_colors


class IPDSim(Model):
    def __init__(self):
        Model.__init__(self)
        self.cfg = load_config()
        cfg = self.cfg

        # game
        self.make_param("rounds", cfg["game"]["rounds"])
        self.make_param("T", float(cfg["game"]["payoffs"]["T"]))
        self.make_param("R", float(cfg["game"]["payoffs"]["R"]))
        self.make_param("P", float(cfg["game"]["payoffs"]["P"]))
        self.make_param("S", float(cfg["game"]["payoffs"]["S"]))
        self.make_param("noise", float(cfg["game"]["noise"]))
        self.make_param("include_self", cfg["ranking"]["include_self"])
        # replicator dynamics
        self.make_param("max_generations", cfg["replicator"]["max_generations"])
        # genetic algorithm
        self.make_param("show_ga", False)
        self.make_param("pop_size", cfg["ga"]["pop_size"])
        self.make_param("mutation_rate", float(cfg["ga"]["mutation_rate"]))
        self.make_param("seed", cfg["seed"])

        self.keys = [s.key for s in Strategies.fixed_field]
        self.tables = [s.table for s in Strategies.fixed_field]
        self.colors = strategy_colors(self.keys, cfg)
        self.reset()

    def current_config(self):
        # the config file with the values currently set in the GUI
        cfg = dict(self.cfg)
        cfg["game"] = {"rounds": int(self.rounds),
                       "payoffs": {"T": self.T, "R": self.R, "P": self.P, "S": self.S},
                       "noise": self.noise}
        cfg["ranking"] = {"include_self": bool(self.include_self)}
        cfg["ga"] = dict(self.cfg["ga"], pop_size=int(self.pop_size),
                         mutation_rate=self.mutation_rate)
        return cfg

    def reset(self):
        cfg = self.current_config()
        self.t = 0
        if self.show_ga:
            self.ga = GeneticAlgorithm(self.tables, cfg, seed=self.seed)
        else:
            result = model.tournament(self.tables, **cfg["game"], **cfg["tournament"],
                                      seed=self.seed)
            self.A = result["average_score"]
            self.scores = model.tournament_scores(self.A, cfg["ranking"]["include_self"])
            run = model.replicator_dynamics(self.A, x0=None,
                                            max_generations=int(self.max_generations),
                                            tolerance=cfg["replicator"]["tolerance"])
            self.history = run["history"]
            self.converged = run["converged"]

    def step(self):
        # returning True tells pyics to stop
        if self.show_ga:
            return self.ga.step()
        if self.t >= len(self.history) - 1:
            return True
        self.t += 1
        return False

    def draw(self):
        plt.cla()
        if self.show_ga:
            self.draw_ga()
        else:
            self.draw_replicator()

    def draw_replicator(self):
        shown = self.history[:self.t + 1]
        for i, key in enumerate(self.keys):
            plt.plot([x[i] for x in shown], label=key, color=self.colors[key],
                     linewidth=2.2 if key == "TFT" else 1.4)
        is_pd = model.is_prisoners_dilemma({"T": self.T, "R": self.R, "P": self.P, "S": self.S})
        title = f"Replicator dynamics, generation {self.t}"
        if not is_pd:
            title += "  (payoffs are not a Prisoner's Dilemma)"
        plt.title(title)
        plt.xlabel("generation")
        plt.ylabel("population share")
        plt.ylim(0, 1)
        plt.legend(ncol=2, fontsize=7, loc="upper right", frameon=False)

    def draw_ga(self):
        h = self.ga.history
        gens = [r["generation"] for r in h]
        plt.plot(gens, [r["best"] for r in h], label="best", color="tab:blue")
        plt.plot(gens, [r["mean"] for r in h], label="mean", color="tab:orange")
        plt.title(f"GA generation {self.ga.generation}, best table {self.ga.best_table}")
        plt.xlabel("generation")
        plt.ylabel("fitness (score per round vs fixed field)")
        plt.legend(frameon=False)


if __name__ == "__main__":
    sim = IPDSim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()