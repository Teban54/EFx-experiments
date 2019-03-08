from Allocation import Allocation
import numpy as np
from expt import generate_utilities


def brute_force(n, m, eval_func, log = True):
    d = np.zeros((m+1, ))
    leximin = Allocation(n, m, eval_matrix)
    flag = False
    while d[0] == 0:
        new_allocation = Allocation(n, m, eval_func)
        for i in range(1, m+1):
            new_allocation.allocate(d[i], i - 1)
        efx, pair = new_allocation.is_EFX()
        if efx and not flag:
            if log:
                print("One possible EFX Allocation is: ")
                print(new_allocation)
            flag = True

        if leximin < new_allocation:
            leximin = new_allocation
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
    lex_efx = leximin.is_EFX()
    if lex_efx:
        if log:
            print("Leximin Allocation is a EFX Allocation")
    else:
        print(eval_func)
        print("Leximin Allocation is not a EFX Allocation")

    return flag


if __name__ == "__main__":
    eval_matrix = np.array([[1, 2, 1], [1, 1, 2], [2, 1, 1]])
    # print(generate_utilities(2, 3, 4))
    flag = True
    N = 5
    M = 8
    cnt = 0
    while flag:
        cnt += 1
        print(cnt)
        eval_matrix = np.random.randint(5, size=(N, M))
        flag = brute_force(N, M, eval_matrix, log=False)
    print(eval_matrix)
