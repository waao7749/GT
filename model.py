"""
model.py - the game, the match, the tournament and the population dynamics.

Contents (Section 1 of the experiment design):
    1. payoffs              get_payoff, is_prisoners_dilemma
    2. one match            play_match
    3. round-robin          tournament
    4. ranking              tournament_scores, ranking
    5. population dynamics  replicator_dynamics

All scores are AVERAGE SCORE PER ROUND, so results with different
numbers of rounds can be compared directly.
"""

import random
from Strategies import C, D, next_move


# ------------------------------------------
# 1. Payoffs
# ------------------------------------------
default_payoffs = {"T":5,  # temptation: I defect while the opponent cooperates
                   "R":3,  # reward: both cooperate
                   "P":1,  # punishment: both defect
                   "S":0}  # sucker: I cooperate while the opponent defects

def get_payoff(my_move, other_move, payoffs):
    if my_move == D and other_move == C:
        return payoffs["T"]
    if my_move == C and other_move == C:
        return payoffs["R"]
    if my_move == D and other_move == D:
        return payoffs["P"]
    return payoffs["S"]

def is_prisoners_dilemma(payoffs):
    T = payoffs["T"]
    R = payoffs["R"]
    P = payoffs["P"]
    S = payoffs["S"]
    return T > R > P > S and (2 * R) > (T + S)

# ------------------------------------------
# 2. One match
# ------------------------------------------
def flip(move):
    # used for noise, create random changes
    if move == C:
        return D
    return C

def play_match(table_a, table_b, rounds = 200, payoffs = None, noise = 0.0, rng = None):
    """Let two rule tables play `rounds` rounds against each other.

    noise: probability that a player's intended move is flipped
           (an execution error, Section 1.3). 0 means no errors.
    rng:   a random.Random object, so results can be reproduced with a seed.

    Returns a dictionary:
        score_a, score_b    average score per round of each player
        coop_a, coop_b      share of rounds in which each player played C
        mutual_coop         share of rounds in which both played C
        moves_a, moves_b    the full move sequences, e.g. "CCDC..."
    """
    if payoffs is None:
        payoffs = default_payoffs
    if rng is None:
        rng = random.Random()

    history_a = []
    history_b = []
    total_a = 0
    total_b = 0
    coop_a = 0
    coop_b = 0
    mutual_coop = 0

    for _ in range(rounds):
        move_a = next_move(table_a, history_a, history_b)
        move_b = next_move(table_b, history_b, history_a)

        # with probability noise the intended move is flipped
        if noise > 0:
            if rng.random() < noise:
                move_a = flip(move_a)
            if rng.random() < noise:
                move_b = flip(move_b)
        #score this round
        total_a += get_payoff(move_a, move_b, payoffs)
        total_b += get_payoff(move_b, move_a, payoffs)
        # count cooperations
        if move_a == C:
            coop_a += 1
        if move_b == C:
            coop_b += 1
        if move_a == C and move_b == C:
            mutual_coop += 1

        history_a.append(move_a)
        history_b.append(move_b)

    return {
        "score_a": total_a / rounds,
        "score_b": total_b / rounds,
        "coop_a": coop_a / rounds,
        "coop_b": coop_b / rounds,
        "mutual_coop": mutual_coop / rounds,
        "moves_a": ''.join(history_a),
        "moves_b": ''.join(history_b),
    }

# ------------------------------------------
# 3. Round-robin tournament
# ------------------------------------------
def empty_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.0)
        matrix.append(row)
    return matrix

def tournament(tables, rounds = 200, payoffs = None, noise = 0.0, repeats = None, seed = None):
    """Round-robin tournament between rule tables, including self-play.

    tables:  list of rule tables (strings), e.g. [s.table for s in fixed_field]
    repeats: how many matches each pair plays. Without noise every match
             gives the same result, so 1 is enough; with noise the default is
             30 and the results are averaged (Section 1.5).
    seed:    random seed, so a noisy tournament can be reproduced.

    Returns a dictionary:
        A       A[i][j] = average score per round of strategy i against j
        mutual  mutual[i][j] = share of rounds with mutual cooperation
        coop    coop[i][j] = share of rounds in which i cooperated against j
    """

    if repeats is None:
        if noise > 0:
            repeats = 30
        else:
            repeats = 1
    rng = random.Random(seed)

    n = len(tables)
    A = empty_matrix(n)
    mutual = empty_matrix(n)
    coop = empty_matrix(n)

    for i in range(n):
        for j in range(i, n): # each pair is counted only once
            sum_i = 0
            sum_j = 0
            sum_coop_i = 0
            sum_coop_j = 0
            sum_mutual =0
            for _ in range(repeats):
                results = play_match(tables[i], tables[j], rounds, payoffs, noise, rng)
                sum_i += results["score_a"]
                sum_j += results["score_b"]
                sum_coop_i += results["coop_a"]
                sum_coop_j += results["coop_b"]
                sum_mutual += results["mutual_coop"]

            if i == j:
                # self-playL both players are the same strategy, so its score is the average of 2 sides
                A[i][i] = (sum_i + sum_j) / (2 * repeats)
                mutual[i][i] = sum_mutual / (2 * repeats)
                coop[i][i] = (sum_coop_i + sum_coop_j) / (2 * repeats)
            else:
                A[i][j] = sum_i / repeats
                A[j][i] = sum_j / repeats
                coop[i][j] = sum_coop_i / repeats
                coop[j][i] = sum_coop_j / repeats
                mutual[i][j] = sum_mutual / repeats
                mutual[j][i] = sum_mutual / repeats

    return {"average_score": A, "mutual_cooperation": mutual, "single-side cooperation": coop}

# ------------------------------------------
# 4. Ranking
# ------------------------------------------
def tournament_scores(A, include_self = False):
    """
    Tournament score S_i: the mean of row i of A
    By default self-play is left out, as in Axelrod's tournament
    """
    scores = []
    for i in range(len(A)):
        total = 0
        count = 0
        for j in range(len(A)):
            if i == j and not include_self:  # skip the game of strategy against itself
                continue
            total += A[i][j]
            count += 1
        scores.append(total / count)
    return scores

def ranking(names, scores):
    pairs = []
    for i in range(len(names)):
        pairs.append((names[i], scores[i]))
    pairs.sort(reverse = True)

    result = []
    rank = 1
    for name, score in pairs:
        result.append((rank, name, score))
        rank += 1
    return result

# -----------------------------------------------
# 5. Replicator dynamics (ecological simulation)
# -----------------------------------------------
def replicator_dynamics(A, x0 = None, max_generations = 2000, tolerance = 1e-8):
    """Follow the population shares x over the generations.

        Each generation:
            f_i     = sum over j of x_j * A[i][j]   (fitness of strategy i)
            f_mean  = sum over i of x_i * f_i       (population average)
            new x_i = x_i * f_i / f_mean

        A:          score matrix from tournament() (self-play included)
        x0:         starting shares; default: every strategy equally common
        tolerance:  stop when the total change sum |new x_i - x_i| is below it

        Returns a dictionary:
            history      list of the share vectors, one per generation
                         (history[0] is the start)
            generations  number of generations that were run
            converged    True if the tolerance was reached
        """
    n = len(A)
    # starting shares
    if x0 is None:
        x = []
        for i in range(n):
            x.append(1.0 / n)
    else:
        total = sum(x0)
        x = []
        for value in x0:
            x.append(value / total)

    # Payoffs must be positive. If the smallest entry of A is 0 or negative,
    # add the same constant to every entry
    smallest = A[0][0]
    for i in range(n):
        for j in range(n):
            if A[i][j] < smallest:
                smallest = A[i][j]
    shift = 0
    if smallest <= 0:
        shift = 1 - smallest  # if the smallest score is negative, then calculate a shift and add it to every score

    history = [list(x)]
    converged = False
    generation = 0

    while generation < max_generations:
        # fitness of strategy i
        fitness = []
        for i in range(n):
            f = 0
            for j in range(n):
                f += (A[i][j] + shift) * x[j]
            fitness.append(f)

        # calculate the average fitness of the whole population
        f_mean = 0
        for i in range(n):
            f_mean += x[i] * fitness[i]

        # calculate the new shares of the next step
        new_x = []
        for i in range(n):
            new_x.append(x[i] * fitness[i] / f_mean)

        # how much did the share change between the 2 consecutive steps
        change = 0
        for i in range(n):
            change += abs(new_x[i] - x[i])

        x = new_x
        history.append(list(x))
        generation += 1

        if change < tolerance:
            converged = True
            break

    return {"history": history, "generations": generation, "converged": converged}

# -----------------------------------------------
# 6. Quick check
# -----------------------------------------------
if __name__ == "__main__":
    import Strategies

    field = Strategies.fixed_field
    names = []
    tables = []
    for s in field:
        names.append(s.key)
        tables.append(s.table)

    # little check for E0
    allc = Strategies.ALLC.table
    alld = Strategies.ALLD.table
    tft = Strategies.TFT.table
    r = play_match(alld, allc, rounds = 200)
    print(f"ALLD VS. ALLC, ALLD score per round: {r['score_a']}, expected to be 5")
    r = play_match(tft, alld, rounds = 200)
    print(f"TFT VS. ALLD, TFT score per round: {r['score_a']}, expected to be 0.995")

    # little check for E1
    result = tournament(tables, rounds = 200)
    scores = tournament_scores(result["average_score"])
    print("\nTournament ranking:")
    for rank, name, score in ranking(names, scores):
        print(f"{rank:2d}. {name:5s} {score:.3f}")

    # little check for E5
    dyn = replicator_dynamics(result["average_score"])
    final = dyn["history"][-1]
    print("\nReplicator dynamics: converged =", dyn["converged"],
          "after", dyn["generations"], "generations")
    for i in range(len(names)):
        print(f"{names[i]:5s} {final[i]:.3f}")