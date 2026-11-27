# The genetic algorith
"""
GA.py - genetic algorithm that evolves a new rule table

The steps follow the assignment:
    0. Encoding        a strategy is its rule table, a string of "C"/"D"
                       (21 rows for memory 2, 5 rows for memory 1)
    1. Initialization  pop_size random rule tables
    2. Selection       every candidate plays a tournament against the fixed
                       field (10 originals + TFT); its fitness is its tournament
                       score, and the top selection_share are kept as parents
    3. Crossover       two random parents are combined row by row
    4. Mutation        every row of a child flips with probability mutation_rate
    5. Repeat          until the best fitness has not improved by more than
                       `tolerance` for `patience` generations, or max_generations

Fitness is the average score per round (as in model.py). With a fixed number
of rounds this ranks candidates exactly like the accumulated payoff that the
assignment mentions.
"""

import itertools
import random

from Strategies import C, D, table_length
from model import play_match


# ------------------------------------------
# Encoding helpers
# ------------------------------------------
def random_table(length, rng):
    # a random rule table, every row C or D with probability 1/2
    return "".join(rng.choice((C, D)) for _ in range(length))

def all_tables(memory):
    length = table_length(memory)
    return ["".join(rows) for rows in itertools.product((C, D), repeat=length)]


# ------------------------------------------
# Fitness
# ------------------------------------------
def candidate_fitness(table, field_tables, game, include_self=True, repeats=None, rng=None):
    """Tournament score of one candidate against the fixed field.

    This is the candidate's row of the tournament matrix: the mean of its
    average scores against every field strategy, plus its own twin when
    include_self is True (the same rule as tournament_scores in model.py).
    """
    if repeats is None:
        repeats = 30 if game["noise"] > 0 else 1
    if rng is None:
        rng = random.Random()

    opponents = list(field_tables)
    if include_self:
        opponents.append(table)

    total = 0.0
    for opp in opponents:
        for _ in range(repeats):
            total += play_match(table, opp, game["rounds"], game["payoffs"],
                                game["noise"], rng)["score_a"]
    return total / (len(opponents) * repeats)


# ------------------------------------------
# The genetic algorithm
# ------------------------------------------
class GeneticAlgorithm:
    """Evolve one rule table against a fixed field.

    Use step() to advance one generation (the GUI does this) or run() to
    go until the stopping rule is met. Every generation is stored in
    self.history so that the fitness curves can be plotted afterwards.
    """

    def __init__(self, field_tables, cfg, seed=None, memory=None):
        ga = cfg["ga"]
        self.field_tables = list(field_tables)
        self.game = cfg["game"]
        self.include_self = cfg["ranking"]["include_self"]
        self.repeats = cfg["tournament"]["repeats"]

        self.pop_size = ga["pop_size"]
        self.n_parents = max(2, round(ga["selection_share"] * ga["pop_size"]))
        self.n_elites = ga["n_elites"]
        self.crossover_mode = ga["crossover"]
        self.mutation_rate = ga["mutation_rate"]
        self.max_generations = ga["max_generations"]
        self.patience = ga["patience"]
        self.tolerance = ga["tolerance"]

        if not 0 <= self.n_elites < self.pop_size:
            raise ValueError("n_elites must be smaller than pop_size")
        if self.crossover_mode not in ("half", "uniform", "fitness"):
            raise ValueError(f"unknown crossover '{self.crossover_mode}'")

        self.memory = cfg["strategy"]["memory"] if memory is None else memory
        self.length = table_length(self.memory)
        self.rng = random.Random(seed)

        self.use_cache = self.game["noise"] == 0
        self.cache = {}

        self.generation = 0
        self.best_table = None
        self.best_fitness = float("-inf")
        self.stale = 0                      # generations without improvement of the best
        self.history = []

        self.population = [random_table(self.length, self.rng) for _ in range(self.pop_size)]         # 1. initialization
        self.fitness = self.evaluate(self.population)
        self.record()

    # -- fitness ---------------------------------------------------------
    def evaluate(self, population):
        scores = []
        for table in population:
            if self.use_cache and table in self.cache:
                scores.append(self.cache[table])
                continue
            f = candidate_fitness(table, self.field_tables, self.game,
                                  self.include_self, self.repeats, self.rng)
            if self.use_cache:
                self.cache[table] = f
            scores.append(f)
        return scores

    def ranked(self):
        return sorted(range(self.pop_size), key=lambda i: self.fitness[i], reverse=True)              # ranked by scores

    # -- variation -------------------------------------------------------
    def crossover(self, a, fa, b, fb):
        if self.crossover_mode == "half":                                                             # first half of the rows from one parent, second half from the other
            cut = self.length // 2
            return a[:cut] + b[cut:]
        if self.crossover_mode == "uniform":
            return "".join(a[i] if self.rng.random() < 0.5 else b[i] for i in range(self.length))     # every row from either parent with probability 1/2
        share_a = fa / (fa + fb) if fa + fb > 0 else 0.5                                              # more weight on fitter parent
        return "".join(a[i] if self.rng.random() < share_a else b[i] for i in range(self.length))     

    def mutate(self, table):
        rows = list(table)
        for i in range(self.length):
            if self.rng.random() < self.mutation_rate:
                rows[i] = D if rows[i] == C else C
        return "".join(rows)

    # -- one generation --------------------------------------------------
    def step(self):
        """Advance one generation. Returns True when the GA has stopped."""
        if self.done():
            return True

        order = self.ranked()
        parents = order[:self.n_parents]                                      # 2. selection
        new_population = [self.population[i] for i in order[:self.n_elites]]  # elites survive unchanged

        while len(new_population) < self.pop_size:
            i, j = self.rng.sample(parents, 2)                                # two different parents
            child = self.crossover(self.population[i], self.fitness[i],
                                   self.population[j], self.fitness[j])       # 3. crossover
            new_population.append(self.mutate(child))                         # 4. mutation

        self.population = new_population
        self.fitness = self.evaluate(self.population)
        self.generation += 1
        self.record()
        return self.done()

    def record(self):
        best = max(range(self.pop_size), key=lambda i: self.fitness[i])
        if self.fitness[best] > self.best_fitness + self.tolerance:
            self.stale = 0
        else:
            self.stale += 1
        if self.fitness[best] > self.best_fitness:
            self.best_fitness = self.fitness[best]
            self.best_table = self.population[best]

        self.history.append({
            "generation": self.generation,
            "best": self.fitness[best],
            "mean": sum(self.fitness) / self.pop_size,
            "diversity": len(set(self.population)),                           # number of different tables
            "best_table": self.population[best],
        })

    def done(self):
        if self.generation >= self.max_generations:
            return True
        if self.generation > 0 and self.stale >= self.patience:
            return True
        else:
            return False

    def run(self):
        while not self.step():
            pass
        return self.result()

    def result(self):
        return {
            "best_table": self.best_table,
            "best_fitness": self.best_fitness,
            "generations": self.generation,
            "converged": self.stale >= self.patience,                         # False if max_generations was hit
            "history": self.history,
            "equals_field_strategy": self.best_table in self.field_tables,
        }


# ------------------------------------------
# Cross-check: exhaustive search for memory 1
# ------------------------------------------
def exhaustive_search(field_tables, cfg, memory=1):
    """Fitness of every possible table (32 for memory 1), best first.

    The GA run with the same memory should find a table whose fitness equals
    the best one here; this is the agreed cross-check for E6.
    """
    scored = []
    for table in all_tables(memory):
        f = candidate_fitness(table, field_tables, cfg["game"],
                              cfg["ranking"]["include_self"], cfg["tournament"]["repeats"])
        scored.append((f, table))
    scored.sort(reverse=True)
    return scored


if __name__ == "__main__":
    import json
    from pathlib import Path
    import Strategies

    cfg = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    field = [s.table for s in Strategies.fixed_field]

    ga = GeneticAlgorithm(field, cfg, seed=cfg["seed"])
    res = ga.run()
    print(f"best fitness {res['best_fitness']:.3f} after {res['generations']} generations "
          f"(converged = {res['converged']})")
    print("best table:", res["best_table"])