import datetime
import random
import numpy as np

N = 3  # Number of players
M = 4  # Number of goods
max_utility = 1.0  # Maximum value of any utility
Ui = []  # List of dicts mapping a subset of goods to its utility


def power_set(n):
    """
    Given a bound n, return the power set of {0,1,...,n-1}, that is, all
    possible sets containing integers up to n-1.
    :param n: Bound on number of elements
    :return: List of frozensets representing the power set (unsorted)
    """
    if n == 0:
        return [frozenset()]
    sets_without = power_set(n-1)
    sets_with = []
    for s in sets_without:
        s1 = set(s)
        s1.add(n-1)
        sets_with.append(frozenset(s1))
    return sets_without + sets_with


def generate_utilities(N, M, max_value):
    """
    Generate random utility functions for each player and each subset of goods.

    Utilities are totally random except they obey the following three rules:
    1) Ui(emptyset) = 0.
    2) 0 < Ui(S) < max_bound for all S that is an nonempty subset of all goods.
    3) If S1 is a subset of S2, then Ui(S1) <= Ui(S2).

    Methodology: (UP FOR DEBATE)
    - For each player, consider all possible subsets of goods (i.e. power set
    of {0,1,...,M-1}), in increasing order of size.
    - For each set of goods, obtain a lower bound on its utility that is the
    maximum of utilities of its subsets, with one good removed.
    - Generate a uniformly random number between this lower bound and
    max_value * (size of set / M), and use it as this set's utility.
        (We use this upper bound instead of max_value, to prevent utilities
        for a single element from being too large)

    Note: This procedure runs in O(N * 2^M * M).

    :param: N: Number of players
    :param: M: Number of goods
    :param: max_value: Maximum value of utility (float)
    :return: List of N dicts, each mapping all possible subset of goods to
        their utility values
    """
    Us = []

    for i in range(N):
        U = {}
        subsets = power_set(M)
        subsets.sort(key=lambda s: len(s))

        for s in subsets:
            if len(s) == 0:
                U[s] = 0
            else:
                # max_bound = max_value
                max_bound = max_value * (len(s) / M)
                min_bound = 0
                for k in s:
                    # Make sure U(s) > U(s\{k})
                    s_remove = set(s)
                    s_remove.remove(k)
                    min_bound = max(min_bound, U[frozenset(s_remove)])
                val = random.uniform(min_bound, max_bound)
                while val == min_bound or val == max_bound:
                    val = random.uniform(min_bound, max_bound)
                U[s] = val

        Us.append(U)

    return Us


def write_utilities_to_file(file_name):
    """
    Write all utilities for each player to an output file.

    Utilities are written in a space-delimited table where rows are subsets of
    goods and columns are players.

    :param file_name: Name of output file
    """
    decimals = 5
    with open(file_name, "w+") as f:
        f.write("Goods")
        for i in range(N):
            f.write(" " + str(i))
        f.write("\n")

        subsets = power_set(M)
        subsets.sort(key=lambda s: len(s))
        for s in subsets:
            if len(s) == 0:
                continue
            f.write("{")
            s_list = list(s)
            for i in range(len(s_list)):
                k = s_list[i]
                f.write((str(k) + ",") if i != len(s_list) - 1 else str(k))
            f.write("}")

            for i in range(N):
                f.write((" %." + str(decimals) + "f") % Ui[i][s])

            f.write("\n")

    print("N = %d, M = %d" % (N, M))
    print("Output written to %s." % file_name)


if __name__ == '__main__':
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
    Ui = generate_utilities(N, M, max_utility)
    write_utilities_to_file(time_str + "_Utilities.txt")
