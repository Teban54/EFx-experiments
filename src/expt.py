import datetime
import random
import numpy as np

N = 3  # Number of players
M = 8  # Number of goods
T = 10 # Number of random iterations for each utility function
max_utility = 1.0  # Maximum value of any utility
Ui = []  # List of dicts mapping a subset of goods to its utility
matrix = None  # Utility matrix


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


def generate_utilities(N, M, max_value, additive, generate_dict=False):
    """
    Generate random utility functions for each player and each subset of goods.

    If additive is true, we assume additive utility. The procedure will
    generate a utility matrix A where A_ij is the value of good j to player i.
    If required (by setting generate_dict to True), it will generate dicts
    of subsets of goods to utilities based on the matrix.

    If not, the procedure generates generic utility functions.
    Utilities are totally random except they obey the following three rules:
    1) Ui(emptyset) = 0.
    2) 0 < Ui(S) < max_bound for all S that is an nonempty subset of all goods.
    3) If S1 is a subset of S2, then Ui(S1) <= Ui(S2).

    Methodology for generic utility functions: (UP FOR DEBATE)
    - For each player, consider all possible subsets of goods (i.e. power set
    of {0,1,...,M-1}), in increasing order of size.
    - For each set of goods, obtain a lower bound on its utility that is the
    maximum of utilities of its subsets, with one good removed.
    - Generate a uniformly random number between this lower bound and
    max_value * (size of set / M), and use it as this set's utility.
        (We use this upper bound instead of max_value, to prevent utilities
        for a single element from being too large)

    Note: This procedure runs in O(N * 2^M * M).

    :param N: Number of players
    :param M: Number of goods
    :param max_value: Maximum value of utility (float)
    :param additive: Whether the additive utility model is used
    :param generate_dict: If additive, whether list of dicts is needed
    :return: - List of N dicts, each mapping all possible subset of goods to
        their utility values
             - Utility matrix, if additive
    """
    if additive:
        A = []
        for i in range(N):
            A.append([])
            for j in range(M):
                A[i].append(random.uniform(0, max_value))

        if generate_dict:
            pass # TODO
        return None, A
    else:
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
    with open(file_name, "a+") as f:
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


def print_utility_matrix(matrix):
    """
    Print the utility matrix.
    """
    print("N = %d, M = %d" % (N, M))
    header = "Player\\Utility"
    decimals = 5
    line = header
    for j in range(M):
        line += " " + str(j) + " " * (decimals + 2 - len(str(j)))
    print(line)

    for i in range(N):
        line = str(i) + " " * (len(header) - len(str(i)))
        for j in range(M):
            line += (" %." + str(decimals) + "f") % matrix[i][j]
        print(line)


def write_utility_matrix_to_file(matrix, file_name):
    """
    Docs TBC
    """
    print("N = %d, M = %d" % (N, M))
    header = "Player\\Utility"
    decimals = 5

    with open(file_name, "a+") as f:
        line = header
        for j in range(M):
            line += " " + str(j) + " " * (decimals + 2 - len(str(j)))
        f.write(line + "\n")

        for i in range(N):
            line = str(i) + " " * (len(header) - len(str(i)))
            for j in range(M):
                line += (" %." + str(decimals) + "f") % matrix[i][j]
            f.write(line + "\n")


# ------------------- Alpha-means --------------------- #

"""
Objective:
    max f(x) = (1/n sum_{i=1 to n} Ui(Si)^alpha) ^ (1/alpha)
where Si is a feasible partition of the goods, Ui is the utility function for
payer i, and alpha is a constant.

This is the Arithmetic Mean when alpha=1, and Harmonic Mean when alpha=-1.
As a special case, if alpha=0, redefine the objective as the Geometric Mean:
    f(x) = (prod_{i=1 to n} Ui(Si)^alpha) ^ (1/n)

The procedure below does the following for each utility function:
1. Generate a random allocation and calculate the objective value.
2. From there, do a local search over allocations by repeatedly trying to 
    allocate a good to someone else if that gives a lower objective value.
3. When a local minimum is found, check if it's EFx and print the allocation.
4. Repeat step 1 on different random allocations T times.
"""


def calc_objective(list, alpha):
    """
    Given a list of utility values, calculate the objective function with a
    single alpha.
    :param list: List of utilities
    :param alpha: Alpha (exponent)
    :return: Objective value
    """
    n = len(list)
    if alpha == 0:
        # Special case: GM
        prod = 1.0
        for u in list:
            prod *= u
        return prod ** (1/n)

    sum = (1/n) * sum([u ** alpha for u in list])
    return sum ** (1/alpha)


def calc_objective(list, alphas, weights):
    """
    Given a list of utility values, calculate the objective function as a
    linear combinaion of sums with different alphas.
    :param list: List of utilities
    :param alphas: List of alphas (exponent)
    :param weights: List of weights for each corresponding alpha
    :return: Objective value
    """
    sum_weights = sum(weights)
    if sum_weights != 0:
        weights = [w / sum_weights for w in weights]
    return sum([calc_objective(list, alphas[i]) * weights[i]
                for i in range(len(alphas))])


def local_search():
    pass


# ------------------- Main --------------------- #

if __name__ == '__main__':
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
    file_name = time_str + ".txt"
    Ui, matrix = generate_utilities(N, M, max_utility, additive=True, generate_dict=False)
    # write_utilities_to_file(file_name)
    print_utility_matrix(matrix)


