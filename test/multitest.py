import numpy as np
import multiprocessing as mp
import os
from time import time

def timetwo(nums):
    print(os.getpid())
    for i,num in enumerate(nums):
        for j in range(10000000):
            j = j+1
            j = j-1
        nums[i] = num*2

    return nums

if __name__ == '__main__':
    np.random.seed(0);

    # create two matrices to be passed
    # to two different processes
    mat1 = np.array([1,2,3])
    mat2 = np.array([6, 5])


        # define number of processes
    ntasks = 6;

    tstart = time()
    # create a pool of processes
    p = mp.Pool(ntasks);

    # feed different process with same task
    # but different data and print the result
    print(p.map(timetwo, [mat1, mat2, mat1]), [mat1, mat2])
    print("parallel time ", time() - tstart)

    # define number of processes
    ntasks = 1;

    tstart = time()
    # create a pool of processes
    p = mp.Pool(ntasks);

    # feed different process with same task
    # but different data and print the result
    print(p.map(timetwo, [mat1, mat2, mat1]), [mat1, mat2])
    print("serial time ", time() - tstart)

