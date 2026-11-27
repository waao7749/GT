"""
experiments.py - E1, E2, E3, E5, E6 and E7.

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


def kendall_tau(a, b):
    # rank the agreement of 2 score ranking lists: +1 same order, 0 unrelated, -1 reversed
    n = len(a)
    concordant, discordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)

def play_all_pairs(tables, rounds):
    """Play every pair once WITHOUT noise and keep the move sequences.

    moves[i][j] = (moves of i, moves of j) in the match i against j.
    Without noise a match is fully determined by the two tables, and the
    tables do not look at the payoffs or at the round number. So:
      - the first N moves of a long match are exactly an N-round match (E2)
      - the moves do not change when the payoff matrix changes (E3)
    We therefore play once and only re-score, which is much faster.
    """

    n = len(tables)
    moves = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            m = model.play_match(tables[i], tables[j], rounds)
            moves[i][j] = (m["moves_a"], m["moves_b"])
            moves[j][i] = (m["moves_b"], m["moves_a"])
    return moves

def outcome_counts(moves, rounds):
    # counts[i][j] = how often (i,j) played CC, CD, DC, DD in the first `rounds` rounds
    n = len(moves)
    counts = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a,b = moves[i][j][0][:rounds], moves[i][j][1][:rounds]
            c = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
            for x, y in zip (a, b):
                c[x + y] += 1
            counts[i][j] = c
    return counts

def matrices_from_counts(counts, payoffs, rounds):
    """Score matrix A and mutual-cooperation matrix from the outcome counts.
    A[i][j] = (R * CC + S * CD + T * DC + P * DD) / rounds, the same average
    score per round that model.tournament returns.
    """
    n = len(counts)
    A = [[0.0] * n for _ in range(n)]
    mutual = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = counts[i][j]
            total = (payoffs["R"] * c["CC"] + payoffs["S"] * c["CD"] + payoffs["T"] * c["DC"] + payoffs["P"] * c["DD"])
            A[i][j] = total / rounds
            mutual[i][j] = c["CC"] / rounds
    return A, mutual

def check_shortcut(tables, cfg, moves):
    rounds = cfg["game"]["rounds"]
    A_fast, _ = matrices_from_counts(outcome_counts(moves, rounds), cfg["game"]["payoffs"], rounds)
    A_slow = score_matrices(tables, cfg, seed = 0)["average_score"]
    diff = max(abs(A_fast[i][j] - A_slow[i][j]) for i in range(len(tables)) for j in range(len(tables)))

    if diff > 1e-9:
        raise RuntimeError(f"re-scoring differs from model.tournament by {diff}!")

def needs_noise_free(cfg, name):
    if cfg["game"].get("noise", 0) > 0:
        raise ValueError("Noise is not 0.0, please check!")

def population_outcome(A, mutual, cfg, threshold):
    # replicator dynamics from the config start: steady-state summary
    run = model.replicator_dynamics(A, **cfg["replicator"])
    final = run["history"][-1]
    return {"final": final,
            "cooperation": population_cooperation(final, mutual),
            "survivors": sum(1 for v in final if v > threshold),
            "winner": max(range(len(final)), key = lambda i: final[i]),
            "converged": run["converged"],
            "generations": run["generations"]
            }

def game_type(T, S):
    if T < 1 and S > 0:
        return "harmony"
    if T > 1 and S > 0:
        return "snowdrift"
    if T < 1 and S < 0:
        return "stag hunt"
    if T > 1 and S < 0:
        return "prisoner's dilemma"
    return "boundary"


# ------------------------------------------
# E1. Baseline Tournament: who is the strongest
# ------------------------------------------
def run_E1(cfg):
    rng = experiment_rng(cfg, 1)
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    result = score_matrices([s.table for s in strategies], cfg, seed = rng.randrange(2 ** 31))
    A, mutual = result["average_score"], result["mutual_cooperation"]

    # ranking without self-play
    scores = model.tournament_scores(A, cfg["ranking"]["include_self"])
    ranking = model.ranking(keys, scores)
    order = [keys.index(name) for _, name, _ in ranking]

    return {"keys": keys,
            "A": A,
            "mutual": mutual,
            "scores": scores,
            "ranking": ranking,
            "order": order,
            "include_self": cfg["ranking"]["include_self"],}


# ------------------------------------------
# E2. Population dynamics: effect of encounters N
# ------------------------------------------
def run_E2(cfg):
    needs_noise_free(cfg, "E2")
    e2 = cfg["experiments"]["E2"]
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    tables = [s.table for s in strategies]
    payoffs = cfg["game"]["payoffs"]
    include_self = cfg["ranking"]["include_self"]

    # play once at largest N, every shorter match is a prefix of it
    rounds_list = sorted(e2["rounds_values"])
    moves = play_all_pairs(tables, max(rounds_list + [cfg["game"]["rounds"]]))
    check_shortcut(tables, cfg, moves)

    per_n = []
    for N in rounds_list:
        A, mutual = matrices_from_counts(outcome_counts(moves, N), payoffs, N)
        scores = model.tournament_scores(A, include_self)
        pop = population_outcome(A, mutual, cfg, e2["survival_threshold"])
        per_n.append({"N": N, "scores": scores, "ranking": model.ranking(keys, scores) , "population": pop})

    # how similar is each ranking to the one at the default N
    A0, _ = matrices_from_counts(outcome_counts(moves, cfg["game"]["rounds"]), payoffs, cfg["game"]["rounds"])
    default_score = model.tournament_scores(A0, include_self)
    for row in per_n:
        row["tau_vs_default"] = kendall_tau(row["scores"], default_score)

    # theory: TFT resists ALLD once N >= (T-P)/(R-P)
    threshold = (payoffs["T"] - payoffs["P"]) / (payoffs["R"] - payoffs["P"])

    # crossover N*: smallest N from which the stable population cooperates most of the time
    crossover = None
    for row in per_n:
        if row["population"]["cooperation"] > 0.5:
            crossover = row["N"]
            break

    return {"keys": keys,
            "per_n": per_n,
            "threshold": threshold ,
            "crossover": crossover,
            "default_rounds": cfg["game"]["rounds"]}


# ------------------------------------------
# E3. Effect of payoff matrices
# ------------------------------------------
def linspace(lo, hi, steps):
    return [lo + (hi - lo) * k / (steps - 1) for k in range(steps)]

def run_E3(cfg):
    needs_noise_free(cfg, "E3")
    e3 = cfg["experiments"]["E3"]
    strategies = field_strategies()
    keys = [s.key for s in strategies]
    tables = [s.table for s in strategies]
    rounds = cfg["game"]["rounds"]
    include_self = cfg["ranking"]["include_self"]

    # the moves do not depend on the payoffs: pl
    moves = play_all_pairs(tables, rounds)
    check_shortcut(tables, cfg, moves)
    counts = outcome_counts(moves, rounds)

    # E3a: (T, S) plane with R = 1 and P = 0
    Ts = linspace(*e3["T_range"], e3["steps"])
    Ss = linspace(*e3["S_range"], e3["steps"])
    cooperation = [[0.0] * len(Ts) for _ in Ss]       # rows: S, columns: T
    survivors = [[0] * len(Ts) for _ in Ss]
    winner = [[""] * len(Ts) for _ in Ss]
    by_type = {}
    for r, S in enumerate(Ss):
        for c, T in enumerate(Ts):
            A, mutual = matrices_from_counts(counts, {"T": T, "R": 1.0, "P": 0.0, "S": S}, rounds)
            pop = population_outcome(A, mutual, cfg, e3["survival_threshold"])
            cooperation[r][c] = pop["cooperation"]
            survivors[r][c] = pop["survivors"]
            winner[r][c] = keys[pop["winner"]]
            by_type.setdefault(game_type(T, S), []).append(pop["cooperation"])
    type_summary = {g: {"cells": len(v), "mean_cooperation": sum(v) / len(v),
                        "min_cooperation": min(v)} for g, v in by_type.items()}

    # E3b: three concrete matrices, compared with the Axelrod ranking
    named = {}
    for name, payoffs in e3["named_payoffs"].items():
        A, mutual = matrices_from_counts(counts, payoffs, rounds)
        scores = model.tournament_scores(A, include_self)
        pop = population_outcome(A, mutual, cfg, e3["survival_threshold"])
        named[name] = {"payoffs": payoffs, "scores": scores,
                       "ranking": model.ranking(keys, scores),
                       "is_pd": model.is_prisoners_dilemma(payoffs),
                       "alternating_beats_mutual_C": payoffs["T"] + payoffs["S"] > 2 * payoffs["R"],
                       "population": pop}
    reference = next(iter(named))
    for name in named:
        named[name]["tau_vs_reference"] = kendall_tau(named[name]["scores"],
                                                      named[reference]["scores"])

    return {"keys": keys, "Ts": Ts, "Ss": Ss, "cooperation": cooperation,
            "survivors": survivors, "winner": winner, "type_summary": type_summary,
            "named": named, "reference": reference}


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