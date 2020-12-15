import os
import time
from mpi4py.futures import MPIPoolExecutor
from concurrent.futures impoer ProcessPollExecutor
from functools import reduce
from ml_kitchen_sink.cv import gaussian, kitchen_sink

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=16) as pool:
        results = pool.map()
