import datetime
import random
import numpy as np
from Allocation2 import Allocation2

N = 3  # Number of players
M = 8  # Number of goods
trials_different_funcs = 300  # Number of experiments with random utility functions
trials_per_func = 10  # Number of random iterations for each utility function
trials_per_search = 1000  # Number of iterations per local search
max_utility = 1.0  # Maximum value of any utility
Ui = None  # List of dicts mapping a subset of goods to its utility
matrix = None  # Utility matrix
additive = True

#alphas = [[1], [0], [-1]]  # List of experiments: each experiment has a number of alpha's with weights
#weights = [[1], [1], [1]]
alphas = []
weights = []
for i in range(41):
    alphas.append([-2 + i * 0.1])
    weights.append([1])

verbose_level = 2

FILE_NAME = ''

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
    header = "Player\\Item"
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
    header = "Player\\Item"
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
    f(x) = (prod_{i=1 to n} Ui(Si)) ^ (1/n)
(To make sure HM is well-defined, we add the constraint that every player is 
allocated at least one good.)

The procedure below does the following for each utility function:
1. Generate a random allocation and calculate the objective value.
2. From there, do a local search over allocations by repeatedly trying to 
    allocate a good to someone else if that gives a lower objective value.
3. When a local minimum is found, check if it's EFx and print the allocation.
4. Repeat step 1 on different random allocations T times.
"""


def calc_objective_single(list, alpha):
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

    obj_sum = (1/n) * sum([u ** alpha for u in list])
    return obj_sum ** (1/alpha)


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
    return sum([calc_objective_single(list, alphas[i]) * weights[i]
                for i in range(len(alphas))])


def local_search(N, M, obj_func, trials=1000, Ui=None, matrix=None, verbose_level=0):
    """
    Perform a single local search.
    :param obj_func: Function that takes in an allocation and returns an
        objective value
    :param trials: Maximum number of iterations
    :param verbose_level: Level of printing logs
    :return: - Value of objective function at local maximum (if found), or
               last iteration (if does not converge)
             - Allocation that achieves the local maximum (as Allocation2 object)
             - Number of iterations for the local search to converge, or
               -1 if never converges
             - Value of maximum objective function among all EFx allocations
               found during the search
             - Allocation that achieves the EFx max (as list of players)
    """
    d = np.random.randint(N, size=M)  # d[i] is the player that item i is allocated to
    all_allocated_one = all([np.count_nonzero(d == i) > 0 for i in range(N)])  # Make sure every player has at least one good
    while not all_allocated_one:
        d = np.random.randint(N, size=M)
        all_allocated_one = all([np.count_nonzero(d == i) > 0 for i in range(N)])

    alloc = Allocation2(N, M, utility_dict=Ui, evaluation_matrix=matrix, allocations=d)
    for i in range(M):
        alloc.allocate(d[i], i)
    obj = obj_func(alloc)
    if verbose_level >= 4:
        print()
        print('      Initial allocation: %s' % str(d))

    best_EFx_obj = -1
    best_EFx_d = None
    if alloc.is_EFX()[0]:
        best_EFx_obj = obj
        best_EFx_d = list(d)
    for t in range(trials):
        local_max = obj
        local_max_item = -1
        local_max_new_agent = -1
        for item in range(M):  # Try allocating one single item to a new agent
            old_agent = d[item]
            if np.count_nonzero(d == old_agent) == 1:  # old_agent only has one item, can't remove
                continue
            for new_agent in range(N):
                if new_agent == old_agent:
                    continue
                alloc.allocate(new_agent, item)
                new_obj = obj_func(alloc)
                if new_obj > local_max:
                    local_max = new_obj
                    local_max_item = item
                    local_max_new_agent = new_agent
            alloc.allocate(old_agent, item)  # Restore to original state

        if local_max == obj:  # Converged
            return obj, alloc, t, best_EFx_obj, best_EFx_d

        if verbose_level >= 4:
            with open(FILE_NAME, 'a+') as f:
                f.write()
                print('      Iteration %d:' % (t+1))
                print('      Reallocated item %d from player %d to %d' % (local_max_item, d[local_max_item], local_max_new_agent))
                print('      Objective value is %.5f' % local_max)

        obj = local_max
        d[local_max_item] = local_max_new_agent
        alloc.allocate(local_max_new_agent, local_max_item)

        if alloc.is_EFX()[0] and best_EFx_obj < obj:
            best_EFx_obj = obj
            best_EFx_d = list(d)

    return obj, alloc, -1, best_EFx_obj, best_EFx_d


def experiment(N, M, alphas, weights, T, Ui=None, matrix=None, verbose_level=0):
    """
    Perform the experiment described above for a fixed utility function.
    :param N: Number of players
    :param M: Number of items
    :param alphas: List of alphas (exponent)
    :param weights: List of weights for each corresponding alpha
    :param T: Number of trials with random initialization
    :param Ui: Utility function as dict, if applicable
    :param matrix: Utility function as matrix, if applicable
    :param verbose_level: Level of printing logs
    :return: - Maximum objective value across T trials
             - Allocation that gives such an objective (as list of players)
             - Whether this allocation is EFx
             - Number of converged trials (out of T)
             - Number of trials that gave an EFx allocation
    """
    def obj_func(alloc):
        """
        Given an allocation, return the objective value.
        :param alloc: Allocation2 object
        :return: Value of objective function
        """
        return calc_objective(alloc.utility_measure(), alphas, weights)

    converged_trials = 0
    EFx_trials = 0
    max = -1
    max_alloc = None

    best_EFx_obj = -1
    best_EFx_d = None
    for t in range(T):
        obj, alloc, num_iter, EFx_obj, EFx_d = local_search(N, M, obj_func, trials_per_search, Ui, matrix, verbose_level)
        converged_trials += 1 if num_iter >= 0 else 0
        EFx_trials += 1 if alloc.is_EFX()[0] else 0
        if obj > max:
            max = obj
            max_alloc = alloc
        if EFx_obj > best_EFx_obj:
            best_EFx_obj = EFx_obj
            best_EFx_d = EFx_d
        if verbose_level >= 3:
            with open(FILE_NAME, 'a+') as f:
                f.write('\n')
                f.write('    Local Search Round #%d:\n' % t)
                f.write('    Objective value = %.5f\n' % obj)
                f.write('    Index of player that each item is allocated to: ' + str(alloc.get_allocation()) + '\n')
                f.write('    This allocation is %sEFx.\n' % ('' if alloc.is_EFX()[0] else 'not '))
                f.write('    Local search ' + ('did not converge.' if num_iter < 0 else 'converged in %d iterations.' % num_iter) + '\n')
                if EFx_obj > -1:
                    f.write('  Maximum objective value of any EFx allocation across all %d local searches: %.5f\n' % (T, EFx_obj))
                    f.write('  Its corresponding EFx allocation : ' + str(EFx_d) + '\n')

    if verbose_level >= 2:
        with open(FILE_NAME, 'a+') as f:
            f.write('\n')
            f.write('  alphas = %s, weights = %s\n' % (alphas, weights))
            f.write('  Maximum objective value = %.5f' % max)
            f.write('  Index of player that each item is allocated to: ' + str(max_alloc.get_allocation()) + '\n')
            f.write('  This allocation is %sEFx.\n' % ('' if max_alloc.is_EFX()[0] else 'not '))
            f.write('  Number of converged trials: %d / %d\n' % (converged_trials, T))
            f.write('  Number of trials that give an EFx allocation: %d / %d\n' % (EFx_trials, T))
            if best_EFx_obj > -1:
                f.write('  Maximum objective value of any EFx allocation across all %d local searches: %.5f\n' % (T, best_EFx_obj))
                f.write('  Its corresponding EFx allocation : ' + str(best_EFx_d) + '\n')
    return max, max_alloc.get_allocation(), max_alloc.is_EFX()[0], converged_trials, EFx_trials


# ------------------- Main --------------------- #

if __name__ == '__main__':
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")
    file_name = time_str + ".txt"
    FILE_NAME = file_name

    EFx_successes = np.zeros(len(alphas))  # Number of EFx trials for each (alpha, weight) set
    most_EFx = -1  # Utility function with most (alpha, weight) pairs being EFx
    most_EFx_util = None
    most_EFx_ties = 0
    most_EFx_index = -1
    least_EFx = 100000000  # Utility function with most (alpha, weight) pairs being EFx
    least_EFx_util = None
    least_EFx_ties = 0
    least_EFx_index = -1
    for t in range(trials_different_funcs):
        if verbose_level >= 0:
            print()
            print('Random utilities #%d' % t)
        Ui, matrix = generate_utilities(N, M, max_utility, additive=additive, generate_dict=False)
        # write_utilities_to_file(file_name)
        with open(file_name, 'a+') as f:
            f.write('Utility matrix %d:\n' % t)
        write_utility_matrix_to_file(matrix, file_name)
        if verbose_level >= 1:
            print_utility_matrix(matrix)

        EFx_members = 0
        for test in range(len(alphas)):
            alpha = alphas[test]
            weight = weights[test]
            max, max_d, is_EFx, converge, EFx_trials = experiment(N, M, alpha, weight, trials_per_func, Ui, matrix, verbose_level)
            EFx_successes[test] += 1 if is_EFx else 0
            EFx_members += 1 if is_EFx else 0

        if EFx_members > most_EFx:
            most_EFx = EFx_members
            most_EFx_util = matrix
            most_EFx_ties = 1
            most_EFx_index = t
        elif EFx_members == most_EFx:
            most_EFx_ties += 1
        if EFx_members < least_EFx:
            least_EFx = EFx_members
            least_EFx_util = matrix
            least_EFx_ties = 1
            least_EFx_index = t
        elif EFx_members == least_EFx:
            least_EFx_ties += 1

    print()
    for test in range(len(alphas)):
        alpha = alphas[test]
        weight = weights[test]
        print('For alphas %s with weights %s, %d / %d results were EFx' % (alphas[test], weights[test], EFx_successes[test], trials_different_funcs))
    print()
    print('%d utility matrices have the most alphas being EFx (%d / %d). One of them is: (#%d)' % (most_EFx_ties, most_EFx, len(alphas), most_EFx_index))
    print_utility_matrix(most_EFx_util)
    print('%d utility matrices have the with least alphas being EFx (%d / %d). One of them is: (#%d)' % (least_EFx_ties, least_EFx, len(alphas), least_EFx_index))
    print_utility_matrix(least_EFx_util)