from Allocation import Allocation
import numpy as np

def brute_force(m, n, eval_func):
    d = np.zeros((n+1, ))
    leximin = Allocation(m, n, eval_matrix)
    flag = False
    while d[0] == 0:
        new_allocation = Allocation(m, n, eval_func)
        for i in range(1, n+1):
            new_allocation.allocate(d[i], i - 1)
        efx, pair = new_allocation.is_EFX()
        if efx and not flag:
            print("One possible EFX Allocation is: ")
            print(new_allocation)
            flag = True

        if leximin < new_allocation:
            leximin = new_allocation
        j = n
        d[j] += 1
        while d[j] == m:
            d[j] = 0
            j -= 1
            d[j] += 1

    if not flag:
        print("There is No EFX Allocation.")

    print("Leximin Allocation is:")
    print(leximin)
    lex_efx = leximin.is_EFX()
    if lex_efx:
        print("Leximin Allocation is a EFX Allocation")
    else:
        print("Leximin Allocation is not a EFX Allocation")

    return flag


if __name__ == "__main__":
    eval_matrix = np.array([[4, 2, 1], [1, 4, 2]])

    brute_force(2, 3, eval_matrix)
