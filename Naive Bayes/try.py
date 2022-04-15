import random
import time
from multiprocessing import Pool
import os
from mpire import WorkerPool
from joblib import Parallel, delayed


# Standart Fonsiyon/Standart Function
def sum_square(number):
    s = 0
    for i in range(number):
        s = i * i
    return s


# Multiprocessing using Process LİB
def sum_square_with_mp(numbers):
    start_time = time.time()
    p = Pool()
    result = p.map(sum_square, numbers)
    p.close()
    p.join()
    end_time = time.time() - start_time
    print(f"Processing {len(numbers)} numbers took {end_time} time using multiprocessing.")


# Serial Processing
def sum_square_no_mp(numbers):
    start_time = time.time()
    result = []
    for i in numbers:
        result.append(sum_square(i))
    end_time = time.time() - start_time
    print(
        f"Processing {len(numbers)} numbers took {end_time} time using serial processing."
    )

def sum_square_with_joblib(numbers):
    results = \
    Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)\
    (delayed(sum_square))
    print(results)


# mpire tools
def sum_square_with_mpire(numbers):
    with WorkerPool(n_jobs=len(numbers)) as pool:
        start_time = time.time()
        result = pool.map(sum_square, numbers)
        end_time = time.time() - start_time
        print(
            f"Processing {len(numbers)} numbers took {end_time} time using mpire processing."
        )
        return end_time


# ray LİB
def sum_square_with_ray(numbers):
    ray.init(num_cpus=5)
    start_time = time.time()
    remote_function = ray.remote(sum_square)
    results = ray.get([remote_function.remote(x) for x in numbers])
    end_time = time.time() - start_time
    print(
        f"Processing {len(numbers)} numbers took {end_time} time using ray  processing."
    )
    return end_time

if __name__ == "__main__":
    number = range(1200)
    number1 = 12000
    sum_square_with_mp(number)
    sum_square_no_mp(number)
    #sum_square_with_mpire(number)
    #sum_square_with_ray(number)
    sum_square_with_joblib(number1)

