import numpy as np

from typing import Callable, List, Iterable
from sys import argv
from time import time
from multiprocessing import Pool
 

def quad_eucliden_distance(left, right):
    sub = left - right
    return np.sum(sub ** 2, axis=1)


def manhatten_distance(left: np.ndarray, right: np.ndarray):
    sub = np.absolute(left - right)
    return np.sum(sub, axis=1)


def knn(
    model: np.ndarray,
    test: np.array,
    distance_f: Callable=quad_eucliden_distance,
    ks: List[int] = [1, 3, 5, 10, 15]
):
    # calculate the distances to all samples for each column
    res = [distance_f(
            model[...,1:],  # model without first column
            np.tile(
                r[1:], (model.shape[0], 1)
            )  # matrix of same size as model with test vector as rows
        ) for r in test]

    return {
        k:
        # take the value that occurs most often in the k nearest neighors
        [np.argmax(np.bincount(r))
            for r in model[
                # find indices of the k nearest neighbors
                [np.argsort(s)[:k] for s in res], 0]
        ] for k in ks
    }


def evaluate(res: Iterable, check: Iterable):
    return sum([1 if a != b else 0 for (a, b) in zip(res, check)])


def main(
    train_filepath: str,
    test_filepath: str,
    n_train: int=5000,
    n_test: int=100
):
    model = np.loadtxt(train_filepath, dtype=int, delimiter=",")
    test = np.loadtxt(test_filepath, dtype=int, delimiter=",")

    np.random.shuffle(model)
    np.random.shuffle(test)

    t_0 = time()

    res = knn(
        model[:n_train,...],
        test[:n_test,...],
        quad_eucliden_distance  # change distance function here
    )

    t_delta = time() - t_0

    print(
        f"{n_train} training samples, tested {n_test} values, time {t_delta}s"
    )

    for (k, v) in res.items():
        n_err = evaluate(v, test[:n_test,0])

        print(f"k={k}:\tmisclassified {n_err}/{n_test}")


if __name__ == "__main__":
    try:
        main(argv[1], argv[2], 26998, 15001)
    except (ValueError, IndexError):
        print(f"Usage: {argv[0]} /path/to/train.csv /path/to/test.csv")