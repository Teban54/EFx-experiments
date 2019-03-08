from Allocation import Allocation
import numpy as np
from expt import generate_utilities


def brute_force(n, m, eval_func, log = True, alpha=1):
    d = np.zeros((m+1, ))
    leximin = Allocation(n, m, eval_matrix)
    max_obj = 0
    max_allocation = Allocation(n, m, eval_matrix)
    flag = False
    epsilon = 0.01
    while d[0] == 0:
        new_allocation = Allocation(n, m, eval_func)
        for i in range(1, m+1):
            new_allocation.allocate(d[i], i - 1)
        efx, pair = new_allocation.is_EFX()

        u = new_allocation.utility_measure()
        u = np.array(u).astype("float")
        u = u + epsilon
        u = np.power(u, alpha)
        obj = np.sum(u)
        if efx and not flag:
            if log:
                print("One possible EFX Allocation is: ")
                print(new_allocation)
            flag = True

        if leximin < new_allocation:
            leximin = new_allocation

        if obj > max_obj:
            max_obj = obj
            max_allocation = new_allocation
        j = m
        d[j] += 1
        while d[j] == n:
            d[j] = 0
            j -= 1
            d[j] += 1

    if not flag and log:
        print("There is No EFX Allocation.")

    if log:
        print("Leximin Allocation is:")
        print(leximin)
    lex_efx, max_pair = leximin.is_EFX()
    max_efx = max_allocation.is_EFX()

    if lex_efx:
        #if log:
            print("Leximin Allocation is a EFX Allocation")
    else:
        #if log:
            print(eval_func)
            print(leximin)
            print(max_pair)
            print(max_allocation)
            print("Leximin Allocation is not a EFX Allocation")

    if max_efx:
        print("Max %f objective is a EFX Allocation" % (alpha,))
    else:
        print(eval_func)
        print(max_allocation)
        print("Max %f objective is not a EFX Allocation" % (alpha,))
        temp = input()
    return flag


if __name__ == "__main__":
    # eval_matrix = np.array([[0.04529, 0.36288, 0.70307, 0.35199, 0.50047], [0.21686, 0.24130, 0.27729, 0.91812, 0.30173], [0.32961, 0.94418, 0.42327, 0.86658, 0.61724]])
    # flag = True
    N = 3
    M = 5
    # a = Allocation(3,5,eval_matrix)
    # a.allocate(0, 2)
    # a.allocate(1, 3)
    # a.allocate(2, 0)
    # a.allocate(2, 1)
    # a.allocate(2, 4)
    # print(a.is_EFX())
    # flag = brute_force(N, M, eval_matrix, log=True)
    # temp = input()
    cnt = 0
    flag = True
    while flag:
        cnt += 1
        epsilon = 0.01
        eval_matrix = np.random.randint(10, size=(N, M)).astype("float") + epsilon
        flag = brute_force(N, M, eval_matrix, log=False, alpha=0.5)
        if cnt % 100 == 0:
            print(eval_matrix)
            print(cnt)
    print(eval_matrix)
