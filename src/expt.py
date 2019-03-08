import numpy as np

N = 3  # Number of players
M = 10  # Number of goods
Ui = []  # List of dicts mapping a subset of goods to its utility


def generate_utilities(N, M):
    """
    Generate random utility functions for each player and each subset of goods.

    Utilities are totally random except they obey the following three rules:
    1) Ui(emptyset) = 0.
    2) Ui(S) > 0 for all S that is an nonempty subset of all goods.
    3) If S1 is a subset of S2, then Ui(S1) <= Ui(S2).

    :param: N: Number of players
    :param: M: Number of goods
    :return: List of N dicts, each mapping all possible subset of goods to
        their utility values
    """
    


if __name__ == '__main__':
    Ui = generate_utilities(N, M)