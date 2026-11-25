"""
rule table encoding and the fixed strategies of the GT lab

A strategy is a string of "C"/"D". With memory length m it
has 1 + 4 + ... + 4**m rows (m = 1 -> 5 rows, m = 2 -> 21 rows):

    row 0       first move(no history)
    row 1..4    second move, indexed by round 1:         (own, opp)
    row 5..20   round 3 onward, indexed by the last 2:   (own t-2, opp t-2, own t-1, own t-2)

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
------------------------------------------------------------
Functions below are for auxiliary purposes
------------------------------------------------------------
"""
from dataclasses import dataclass, field
import random
import warnings

C, D = (1, 2)
moves = (C, D)

def table_length(m):
    return sum(4 ** k for k in range(m + 1))

memory_of_length = {table_length(m): m for m in (1, 2)} # table length -> memory length

def block_offset(k):
    # index of the first row of the block that uses k rounds of history
    return sum(4 ** j for j in range(k))



"""
------------------------------------------------------------
Strategy definition and functions
------------------------------------------------------------
opp and own are 2 tuples restoring the latest 2 actions of myself and the opponent
    - len(opp) = 0 -> first round, no history
    - len(opp) = 1 -> opp = (C), own = (D)
    - len(opp) = 2 -> opp = (C,D), own = (D,D)
(actions input are just for examples)
"""
def lenient(opp):
    """
    1. Cooperate in the first 2 rounds
    retaliates only after 2 consecutive rounds of defection
    """
    if len(opp) < 2:
        return C
    return D if opp[-1] == C and opp[-2] == D else C

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
    if not opp:
        return C
    if opp[-2] == C and own[-2] == D:
        return C
    return opp[-1]  # return to TFT

def pavlov_2(own, opp):
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
    if not opp: # first round
        return C
    if len(opp) == 1:
        return D
    if own[-1] == D and opp[-1] == C and opp[-2] == C:
        return D
    if own[-1] == D and opp[-2] == C and opp[-1] == D:
        return C
    return opp[-1]
