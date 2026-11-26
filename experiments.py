# functions for graph E0-E8

"""
experiments.py - E5, E6 and E7.

    E5  population dynamics   replicator dynamics from a uniform start and
                              from random starts; the stable state (section c)
    E6  three criteria        initial viability, robustness and stability
                              (the evaluation criteria listed in the assignment)
    E7  genetic algorithm     evolve a new strategy against the fixed field,
                              plus the memory-1 exhaustive cross-check

Every run_E* function returns a plain dictionary; every fig_E* function draws
one figure on a given matplotlib Axes, so make_figures.py can combine them.
All values come from config.json, and every random generator is seeded from
cfg["seed"] plus the experiment number, so results can be reproduced.

Usage:  python experiments.py E5      (or E6, E7, all)
"""

import json
import random
import sys
from pathlib import Path

import Strategies
import model
from GA import GeneticAlgorithm, candidate_fitness, exhaustive_search, random_table

CONFIG_PATH = Path(__file__).with_name("config.json")


# ------------------------------------------
# Shared helpers
# ------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def experiment_rng(cfg, number):
    # one independent, reproducible generator per experiment
    return random.Random(cfg["seed"] * 100 + number)

def field_strategies():
    return list(Strategies.fixed_field)

def score_matrices(tables, cfg, seed):
    # tournament with the config settings; self-play is always played
    return model.tournament(tables, **cfg["game"], **cfg["tournament"], seed=seed)

def random_shares(n, rng):
    # a random starting composition, uniform over all compositions (Dirichlet(1, ..., 1))
    w = [rng.expovariate(1.0) for _ in range(n)]
    total = sum(w)
    return [v / total for v in w]

def population_cooperation(x, mutual):
    # share of all encounters in the population that are mutual cooperation
    n = len(x)
    return sum(x[i] * x[j] * mutual[i][j] for i in range(n) for j in range(n))

def strategy_colors(keys, cfg):
    # fixed colour per strategy, TFT in black (config), the rest from tab10
    import matplotlib.pyplot as plt
    fixed = cfg["figures"]["colors"]
    cycle = plt.get_cmap("tab10").colors
    colors, k = {}, 0
    for key in keys:
        if key in fixed:
            colors[key] = fixed[key]
        else:
            colors[key] = cycle[k % len(cycle)]
            k += 1
    return colors


# ------------------------------------------
# E5. Population dynamics
# ------------------------------------------
def run_E5(cfg):
    rng = experiment_rng(cfg, 5)
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    result = score_matrices([s.table for s in strategies], cfg, seed=rng.randrange(2**31))
    A, mutual = result["average_score"], result["mutual_cooperation"]

    # uniform start (the default of the model and the GUI)
    uniform = model.replicator_dynamics(A, **cfg["replicator"])
    uniform["cooperation"] = [population_cooperation(x, mutual) for x in uniform["history"]]

    # random starts: does the stable state depend on where the population begins?
    rep = dict(cfg["replicator"])
    rep.pop("x0")
    random_runs = []
    for _ in range(cfg["experiments"]["E5"]["n_random_starts"]):
        x0 = random_shares(len(keys), rng)
        run = model.replicator_dynamics(A, x0=x0, **rep)
        random_runs.append({"x0": x0, "final": run["history"][-1],
                            "generations": run["generations"], "converged": run["converged"]})

    return {"keys": keys, "A": A, "uniform": uniform, "random_runs": random_runs}

def fig_E5(data, ax, cfg):
    # shares over the generations from the uniform start
    colors = strategy_colors(data["keys"], cfg)
    history = data["uniform"]["history"]
    for i, key in enumerate(data["keys"]):
        ax.plot([x[i] for x in history], label=key, color=colors[key],
                linewidth=2.2 if key == "TFT" else 1.4)
    ax.set_xlabel("generation")
    ax.set_ylabel("population share")
    ax.set_title("E5: replicator dynamics from a uniform start")
    ax.legend(ncol=2, fontsize=8, frameon=False)


# ------------------------------------------
# E6. Initial viability, robustness, stability
# ------------------------------------------
def initial_viability(A, keys, invader_share, cfg):
    """Can strategy i get a foothold when it starts rare?

    The other strategies share the rest of the population equally. A
    strategy is viable when its fitness at the start beats the population
    mean (growth factor > 1), i.e. its share grows in the first generation.
    The final share after the full replicator run is reported as well.
    """
    n = len(A)
    rep = dict(cfg["replicator"])
    rep.pop("x0")
    out = {}
    for i, key in enumerate(keys):
        x0 = [(1 - invader_share) / (n - 1)] * n
        x0[i] = invader_share
        fitness = [sum(A[a][b] * x0[b] for b in range(n)) for a in range(n)]
        mean = sum(x0[a] * fitness[a] for a in range(n))
        final = model.replicator_dynamics(A, x0=x0, **rep)["history"][-1][i]
        out[key] = {"growth": fitness[i] / mean, "final_share": final}
    return out

def viability_against_alld(tables, keys, invader_share, cfg, rng):
    """Can strategy i invade a world of defectors (ALL D)?

    This is the 'asocial world' of Axelrod & Hamilton (1981): a small group
    of i entering a population of ALL D. Growth > 1 means it can start.
    """
    alld = Strategies.ALLD.table
    out = {}
    for table, key in zip(tables, keys):
        A = score_matrices([table, alld], cfg, seed=rng.randrange(2**31))["average_score"]
        x = [invader_share, 1 - invader_share]
        f_i = A[0][0] * x[0] + A[0][1] * x[1]
        f_d = A[1][0] * x[0] + A[1][1] * x[1]
        mean = x[0] * f_i + x[1] * f_d
        out[key] = {"growth": f_i / mean}
    return out

def robustness(A, tables, keys, cfg, rng):
    """Does strategy i do well in many different environments?

    1. random opponents: mean score against random rule tables
    2. random starts:    in how many replicator runs from random starting
                         shares strategy i survives (final share > threshold)
    3. leave one out:    best and worst rank when one other strategy is removed
    """
    e6 = cfg["experiments"]["E6"]
    include_self = cfg["ranking"]["include_self"]
    n = len(A)
    length = len(tables[0])

    opponents = [random_table(length, rng) for _ in range(e6["n_random_opponents"])]
    vs_random = [candidate_fitness(t, opponents, cfg["game"], include_self=False,
                                   repeats=cfg["tournament"]["repeats"], rng=rng)
                 for t in tables]

    rep = dict(cfg["replicator"])
    rep.pop("x0")
    survived = [0] * n
    for _ in range(e6["n_random_starts"]):
        final = model.replicator_dynamics(A, x0=random_shares(n, rng), **rep)["history"][-1]
        for i in range(n):
            if final[i] > e6["survival_threshold"]:
                survived[i] += 1

    rank_range = []
    for i in range(n):
        ranks = []
        for removed in range(n):
            if removed == i:
                continue
            keep = [k for k in range(n) if k != removed]
            sub = [[A[a][b] for b in keep] for a in keep]
            scores = model.tournament_scores(sub, include_self)
            order = sorted(range(len(keep)), key=lambda k: scores[k], reverse=True)
            ranks.append(order.index(keep.index(i)) + 1)
        rank_range.append((min(ranks), max(ranks)))

    return {key: {"vs_random_tables": vs_random[i],
                  "survived": survived[i],
                  "rank_best": rank_range[i][0],
                  "rank_worst": rank_range[i][1]}
            for i, key in enumerate(keys)}

def stability(A, keys, tol):
    """Once established, can strategy i resist each rare invader j?

    Evolutionary stability (Axelrod & Hamilton 1981): a resident i resists a
    rare invader j if A[i][i] > A[j][i], or if those are equal and
    A[i][j] > A[j][j]. If both pairs are equal, j is neutral: it neither
    grows nor shrinks and can drift in.
    """
    n = len(A)
    out = {}
    for i, key in enumerate(keys):
        resisted, neutral, invaded_by = [], [], []
        for j in range(n):
            if j == i:
                continue
            if A[i][i] > A[j][i] + tol:
                resisted.append(keys[j])
            elif abs(A[i][i] - A[j][i]) <= tol:
                if A[i][j] > A[j][j] + tol:
                    resisted.append(keys[j])
                elif abs(A[i][j] - A[j][j]) <= tol:
                    neutral.append(keys[j])
                else:
                    invaded_by.append(keys[j])
            else:
                invaded_by.append(keys[j])
        out[key] = {"resisted": resisted, "neutral": neutral, "invaded_by": invaded_by}
    return out

def run_E6(cfg):
    rng = experiment_rng(cfg, 6)
    e6 = cfg["experiments"]["E6"]
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    tables = [s.table for s in strategies]
    A = score_matrices(tables, cfg, seed=rng.randrange(2**31))["average_score"]

    return {
        "keys": keys,
        "viability_field": initial_viability(A, keys, e6["invader_share"], cfg),
        "viability_alld": viability_against_alld(tables, keys, e6["invader_share"], cfg, rng),
        "robustness": robustness(A, tables, keys, cfg, rng),
        "stability": stability(A, keys, e6["tie_tolerance"]),
        "n_random_starts": e6["n_random_starts"],
    }

def fig_E6(data, ax, cfg):
    # one panel: survival frequency (robustness) against growth when rare (viability)
    colors = strategy_colors(data["keys"], cfg)
    for key in data["keys"]:
        x = data["viability_field"][key]["growth"]
        y = data["robustness"][key]["survived"] / data["n_random_starts"]
        ax.scatter(x, y, color=colors[key], s=40)
        ax.annotate(key, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("growth factor when rare (initial viability)")
    ax.set_ylabel("share of random starts survived (robustness)")
    ax.set_title("E6: viability and robustness")


# ------------------------------------------
# E7. Genetic algorithm
# ------------------------------------------
def run_E7(cfg):
    rng = experiment_rng(cfg, 7)
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    field_tables = [s.table for s in strategies]

    # several independent GA runs, each with its own seed
    runs = []
    for _ in range(cfg["ga"]["n_runs"]):
        ga = GeneticAlgorithm(field_tables, cfg, seed=rng.randrange(2**31))
        runs.append(ga.run())
    best = max(runs, key=lambda r: r["best_fitness"])

    # how does the evolved strategy do inside the field?
    tables = field_tables + [best["best_table"]]
    names = keys + ["GA"]
    result = score_matrices(tables, cfg, seed=rng.randrange(2**31))
    A = result["average_score"]
    scores = model.tournament_scores(A, cfg["ranking"]["include_self"])
    dynamics = model.replicator_dynamics(A, **cfg["replicator"])

    out = {
        "runs": runs,
        "best_table": best["best_table"],
        "best_fitness": best["best_fitness"],
        "equals_field_strategy": best["equals_field_strategy"],
        "ranking": model.ranking(names, scores),
        "final_shares": dict(zip(names, dynamics["history"][-1])),
    }

    # cross-check: the GA with memory 1 must reach the best of all 32 tables
    if cfg["experiments"]["E7"]["check_memory1_exhaustive"]:
        exhaustive = exhaustive_search(field_tables, cfg, memory=1)
        ga_best = [GeneticAlgorithm(field_tables, cfg, seed=rng.randrange(2**31), memory=1).run()
                   for _ in range(cfg["experiments"]["E7"]["memory1_runs"])]
        out["memory1_check"] = {
            "exhaustive_best": exhaustive[0],
            "ga_best": [(r["best_fitness"], r["best_table"]) for r in ga_best],
            "passed": all(abs(r["best_fitness"] - exhaustive[0][0]) < 1e-9 for r in ga_best),
        }
    return out

def fig_E7(data, ax, cfg):
    # best and mean fitness per generation for every run
    for k, run in enumerate(data["runs"]):
        gens = [h["generation"] for h in run["history"]]
        ax.plot(gens, [h["best"] for h in run["history"]], color="tab:blue", alpha=0.6,
                linewidth=1, label="best" if k == 0 else None)
        ax.plot(gens, [h["mean"] for h in run["history"]], color="tab:orange", alpha=0.4,
                linewidth=1, label="mean" if k == 0 else None)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (score per round vs fixed field)")
    ax.set_title(f"E7: genetic algorithm, {len(data['runs'])} runs")
    ax.legend(frameon=False)


# ------------------------------------------
# Command line
# ------------------------------------------
def save_figure(draw, data, cfg, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(__file__).with_name(cfg["figures"]["out_dir"])
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    draw(data, ax, cfg)
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=cfg["figures"]["dpi"])
    plt.close(fig)
    return path

def report_E5(data):
    u = data["uniform"]
    print(f"E5 uniform start: converged = {u['converged']} after {u['generations']} generations")
    for key, share in zip(data["keys"], u["history"][-1]):
        print(f"   {key:4s} {share:.3f}")
    print(f"   population mutual cooperation: {u['cooperation'][-1]:.3f}")
    not_conv = sum(not r["converged"] for r in data["random_runs"])
    print(f"   random starts: {len(data['random_runs'])} runs, {not_conv} did not converge")

def report_E6(data):
    print("E6  key  growth(field) growth(ALLD) vsRandom survived rank   resisted/neutral/invaded")
    for key in data["keys"]:
        v, a = data["viability_field"][key], data["viability_alld"][key]
        r, s = data["robustness"][key], data["stability"][key]
        print(f"    {key:4s} {v['growth']:.3f}         {a['growth']:.3f}        "
              f"{r['vs_random_tables']:.3f}    {r['survived']:2d}/{data['n_random_starts']}  "
              f"{r['rank_best']}-{r['rank_worst']}  "
              f"{len(s['resisted'])}/{len(s['neutral'])}/{len(s['invaded_by'])}")

def report_E7(data):
    print(f"E7 best table {data['best_table']}  fitness {data['best_fitness']:.3f}  "
          f"(same as a field strategy: {data['equals_field_strategy']})")
    for rank, name, score in data["ranking"][:5]:
        print(f"   {rank:2d}. {name:4s} {score:.3f}")
    print(f"   final share of GA strategy: {data['final_shares']['GA']:.3f}")
    if "memory1_check" in data:
        m1 = data["memory1_check"]
        print(f"   memory-1 check passed = {m1['passed']}  exhaustive best {m1['exhaustive_best']}")

if __name__ == "__main__":
    cfg = load_config()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"E5": (run_E5, fig_E5, report_E5),
            "E6": (run_E6, fig_E6, report_E6),
            "E7": (run_E7, fig_E7, report_E7)}
    for name in (jobs if which == "all" else [which]):
        run, fig, report = jobs[name]
        data = run(cfg)
        report(data)
        print("   figure:", save_figure(fig, data, cfg, name))