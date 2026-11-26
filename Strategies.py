"""
rule table encoding and the fixed strategies of the GT lab

A strategy is a string of "C"/"D". With memory length m it
has 1 + 4 + ... + 4**m rows (m = 1 -> 5 rows, m = 2 -> 21 rows):

    row 0       first move(no history)
    row 1..4    second move, indexed by round 1:         (own, opp)
    row 5..20   round 3 onward, indexed by the last 2:   (own t-2, opp t-2, own t-1, opp t-1)

Inside each block the history is read as a binary number with C = 0 and
D = 1, oldest round first, own move before the opponent's move.
Example (m = 2): own t-2 = C, opp t-2 = D, own t-1 = D, opp t-1 = C
-> bits 0 1 1 0 -> 6 -> row 5 + 6 = 11.

A memory-1 table (5 rows) is read the same way with only one round of
history, so model.py can let tables of different length play each other.

Every fixed strategy below is first written as a readable rule function
rule(own, opp) -> 'C' | 'D', where own / opp hold at most the last two
moves (oldest first). build_table() then enumerates every possible history
and turns the rule into its 21-row table, so the code and the table can
never disagree.
"""

"""
--------------------------------------------
Supporting functions
--------------------------------------------
"""
from dataclasses import dataclass, field

C, D = ("C", "D")
moves = (C, D)

def table_length(m):
    return sum(4 ** k for k in range(m + 1))

memory_of_length = {table_length(m): m for m in (1, 2)} # table length -> memory length

def block_offset(k):
    # index of the first row of the block that uses k rounds of history
    return sum(4 ** j for j in range(k))

def _pair_bits(own_move, opp_move):  # code actions of myself and opp into binary, C=0 and D=1
    return (2 if own_move == "D" else 0) + (1 if opp_move == "D" else 0)

def row_index(own_hist, opp_hist, m):
    k = min(len(own_hist), m)
    idx = 0
    for j in range(len(own_hist) - k, len(own_hist)):
        idx = idx * 4 + _pair_bits(own_hist[j], opp_hist[j])
    return block_offset(k) + idx

def decode_row(row, m):
    # inverse of row_index: the (own, opp) history that a row stands for
    k = 0
    while row >= block_offset(k + 1):
        k += 1
    if k > m:
        raise ValueError(f"row {row} does not exist for memory length {m}!")

    idx = row - block_offset(k)
    own, opp = [], []
    for i in range(k):
        divisor = 4 ** (k - 1 - i)
        pair = idx // divisor
        idx = idx % divisor
        if pair == 0:    # 0 -> (C,C), defined in function _pair_bits
            own.append(C)
            opp.append(C)
        elif pair == 1:  # 1 -> (C,D)
            own.append(C)
            opp.append(D)
        elif pair == 2:  # 2 -> (D,C)
            own.append(D)
            opp.append(C)
        else:            #3 -> (D,D)
            own.append(D)
            opp.append(D)
    return tuple(own), tuple(opp)


def next_move(table, own_hist, opp_hist):
    # Move prescribed by a rule table, given both players' full histories
    m = memory_of_length[len(table)]
    return table[row_index(own_hist, opp_hist, m)]

def build_table(rule, m = 2):
    # Turn rule function into its rule table by enumerating all histories
    rows = []
    for row in range(table_length(m)):
        own, opp = decode_row(row, m)  # change the row numbers into action history
        move = rule(own, opp)
        if move not in moves:
            raise ValueError(f"rule returned {move} for history {own} / {opp}!")
        rows.append(move)
    return "".join(rows)

def format_table(table):
    m = memory_of_length[len(table)]
    lines = []
    for row, move in enumerate(table):
        own, opp = decode_row(row, m)
        history = " ".join(o + p for o, p in zip(own, opp)) or "start"
        lines.append(f"{row:2d} {history:<7} -> {move}")

    return "\n".join(lines)



"""
-----------------------------------
Strategy definition and functions
-----------------------------------
Reference Strategies
"""

@dataclass(frozen = True)
class Strategy:
    key: str
    name: str
    table: str
    summary: str
    traits: tuple = field(default = ())


def make_strategy(key, name, rule, summary, traits = (), m = 2):
    return Strategy(key, name, build_table(rule, m), summary, tuple(traits))

def tft(own, opp):
    return C if not opp else opp[-1]

def allc(own, opp):
    return C

def alld(own, opp):
    return D

TFT = make_strategy(
    "TFT", "Tit for Tat", tft,
    "cooperates first, then copies the opponent's previous move",
    ("nice", "retaliatory", "forgiving")
)
ALLC = make_strategy(
    "ALLC", "Always cooperate", allc,
    "Always cooperates",
    ("nice", "forgiving")
)
ALLD = make_strategy(
    "ALLD", "Always defect", alld,
    "Always defects",
    ("exploitative",)
)


"""
Competitor strategies
opp and own are 2 tuples restoring the latest 2 actions of myself and the opponent
    - len(opp) = 0 -> first round, no history
    - len(opp) = 1 -> opp = (C), own = (D)
    - len(opp) = 2 -> opp = (C,D), own = (D,D)
(actions input are just for examples)
"""
def lenient(own, opp):
    """
    1. Cooperate in the first 2 rounds
    retaliates only after 2 consecutive rounds of defection
    """
    if len(opp) < 2:
        return C
    return D if opp[-1] == D and opp[-2] == D else C

def double_punisher(own, opp):
    """
    2. Punishes being exploited twice, mutual defection only once.
    - The opponent defected last round -> defect.
    - Two rounds ago the opponent defected while I cooperated (I was the
      sucker) -> defect again, a second punishment.
    - Two rounds ago we both defected and the opponent has now cooperated
      -> accept the peace offer and cooperate.
    """
    if not opp:
        return C
    if opp[-1] == D:
        return D
    if len(opp) == 2 and own[-2] == C and opp[-2] == D:
        return D
    return C

def contrite(own, opp):
    """
    3. TFT that does not answer retaliation it provoked itself.

    If two rounds ago I defected while the opponent cooperated, the
    opponent's defection last round is a reply to me, so I cooperate instead
    of retaliating. This ends the C/D echo that locks two TFT players
    after a single defection.
    """
    if not opp: # first round
        return C
    if len(opp) == 2 and own[-2] == D and opp[-2] == C and opp[-1] == D:
        return C
    return opp[-1]  # return to TFT

def pavlov2(own, opp):
    """
    4. Win-stay, lose shift with patience

    A round counts as a win when the opponent cooperated (payoff R or T)
    and as a loss when the opponent defected (payoff S or P); under the
    Prisoner's dilemma ordering this matches "payoff >= R". This strategy
    keeps its move and switches only after 2 losses in a round that both
    came after its last switch, so it never changes its move twice running.
    """
    if not own:
        return C
    if len(own) < 2:
        return own[-1]

    # I lose when (D,D) and (C,D) for (I,opp)
    two_losses = opp[-1] == D and opp[-2] == D
    kept_move = own[-1] == own[-2]
    if two_losses and kept_move:
        return D if own[-1] == C else C
    return own[-1]

def prober(own, opp):
    """
    5.  Tests the opponent in round 2 and exploits it if it does not react

    - Round 1: cooperate. Round 2: defect
    - Keep exploiting while the opponent cooperated in both of the last two
      rounds though I defected last time
    - If the opponent answered an exploitation with a defection (I defected
      two rounds ago while it cooperated, then it defected), apologize with a
      cooperation and fall back to TFT.
    - Otherwise play TFT
    """
    if not opp:
        return C
    if len(opp) == 1:
        return D
    if own[-1] == D and opp[-1] == C and opp[-2] == C:
        return D
    if own[-2] == D and opp[-2] == C and opp[-1] == D:
        return C
    return opp[-1]

def _mutual_c(own, opp, i):
    # True if both players cooperated in round i (negative index, e.g. -1 = last round)
    return own[i] == C and opp[i] == C

def _mutual_d(own, opp, i):
    # True if both players defected in round i
    return own[i] == D and opp[i] == D

def suspicious(own, opp):
    """
    6.
    - Round 1: defect (hostile opening)
    - Two rounds of mutual defection in a row -> cooperate once to break
      the (D,D) lock
    - Otherwise play TFT
    """
    if not opp:
        return D
    if len(opp) == 2 and _mutual_d(own, opp, -1) and _mutual_d(own, opp, -2):
        return C
    return opp[-1]

def strict_mutualist(own, opp):
    """
    7.
    - Round 1: cooperate (no history yet)
    - Round 2: only one round has been played, so that round alone must
      be mutual cooperation (C,C)
    - Round 3 onward: both of the last two rounds must be (C,C)
    - Otherwise defect. 
    """
    if not opp:
        return C
    if len(opp) == 1:
        return C if _mutual_c(own, opp, -1) else D
    if _mutual_c(own, opp, -1) and _mutual_c(own, opp, -2):
        return C
    return D

def two_round_grudger(own, opp):
    """
    8.
    - Round 1: cooperate
    - Defect if the opponent defected at least once in the last two rounds,
      otherwise cooperate
    """
    return D if D in opp else C

def opportunist(own, opp):
    """
    9.
    - Round 1: cooperate
    - After two rounds of mutual cooperation, defect once by surprise
    - Otherwise play TFT
    """
    if not opp:
        return C
    if len(opp) == 2 and _mutual_c(own, opp, -1) and _mutual_c(own, opp, -2):
        return D
    return opp[-1]

def contrarian(own, opp):
    """
    10.
    - Round 1: cooperate
    - Afterwards: opponent C -> D, opponent D -> C
    """
    if not opp:
        return C
    return D if opp[-1] == C else C

LENIENT = make_strategy(
    "LEN", "Lenient", lenient,
    "Retaliates only after 2 consecutive defections",
    ("nice", "retaliatory", "forgiving")
)

DOUBLE_PUNISHER = make_strategy(
    "DPN", "Double Punisher", double_punisher,
    "Defects twice after being exploited, once after mutual defection",
    ("nice", "retaliatory", "forgiving")
)

CONTRITE = make_strategy(
    "CTR", "Contrite", contrite,
    "TFT, but accepts the retaliation provoked by its own defection",
    ("nice", "retaliatory", "forgiving")
)

PAVLOV2 = make_strategy(
    "PAV", "Pavlov-2", pavlov2,
    "Keeps its move after a win (payoff = R or T), switches only after 2 consecutive losses",
    ("nice", "retaliatory", "self-correcting")
)

PROBER = make_strategy(
    "PRB", "Prober", prober,
    "Tests the opponent in round 2 and exploits it if it does not react, otherwise apologize and play TFT",
    ("exploitative", "retaliatory", "forgiving")
)

SUSPICIOUS = make_strategy(
    "SUS", "Suspicious", suspicious,
    "Defects in round 1, then plays TFT; cooperates to break a deadlock after 2 rounds of mutual defection",
    ("not nice", "retaliatory", "forgiving")
)

STRICT_MUTUALIST = make_strategy(
    "SMU", "Strict Mutualist", strict_mutualist,
    "Cooperates only if both players cooperated in both of the last 2 rounds",
    ("nice", "retaliatory", "unforgiving")
)

TWO_ROUND_GRUDGER = make_strategy(
    "GR2", "Two-round Grudger", two_round_grudger,
    "Defects if the opponent defected at least once in the last 2 rounds",
    ("nice", "strongly retaliatory")
)

OPPORTUNIST = make_strategy(
    "OPT", "Opportunist", opportunist,
    "After 2 rounds of mutual cooperation defects once by surprise, otherwise plays TFT",
    ("nice opening", "exploitative")
)

CONTRARIAN = make_strategy(
    "CRN", "Contrarian", contrarian,
    "Plays the opposite of the opponent's last move",
    ("control",)
)

originals = [LENIENT, DOUBLE_PUNISHER, CONTRITE, PAVLOV2, PROBER,
             SUSPICIOUS, STRICT_MUTUALIST, TWO_ROUND_GRUDGER, OPPORTUNIST, CONTRARIAN]

# TFT is the benchmark and competes in the field.
# ALLC and ALLD never compete; only used to verify the model (E0).
# The lists do not overlap, so no strategy can appear twice when they are joined.
benchmark = TFT
references = [ALLC, ALLD]
fixed_field = originals + [benchmark]  # the 10 originals + TFT, used in E1-E7

# every strategy exactly once; use this for look-ups, menus (GUI) and checks
all_strategies = fixed_field + references
by_key = {s.key: s for s in all_strategies}

def get(key):
    # look up a strategy by key
    return by_key[key]

def check_registry():
    # check if 2 strategies share a key or a rule table
    strategies = originals + references
    keys = [s.key for s in strategies]
    if len(set(keys)) != len(keys):
        raise ValueError ("duplicate strategy keys")

    # the fixed field must hold every strategy once and contain the benchmark;
    field_keys = [s.key for s in fixed_field]
    if len(set(field_keys)) != len(field_keys):
        raise ValueError ("fixed_field contains the same strategy twice")
    if field_keys.count(benchmark.key) != 1:
        raise ValueError (f"fixed_field must contain {benchmark.key} exactly once")

    seen = {}
    for s in strategies:
        if len(s.table) != table_length(2):
            raise ValueError (f"{s.key}: table has {len(s.table)} rows, expected to be 21")
        if s.table in seen:
            raise ValueError (f"{s.key} has the same rule table as {seen[s.table]}")
        seen[s.table] = s.key  # store the seen table and assigned to the strategy it belongs to

check_registry()

if __name__ == "__main__":
    for m in (1, 2):
        for row in range(table_length(m)):
            own, opp = decode_row(row, m)
            assert row_index(own, opp, m) == row
    print("row -> history -> row round-trip check OK")
    for s in fixed_field:
        print(f"\n{s.key} {s.name} ({','.join(s.traits)})")
        print(s.table)
        print(format_table(s.table))