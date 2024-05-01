from numpy.random import default_rng
import numpy as np

matrices = {

'small_blocks': np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=int),


'reciprocal': np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=int),

}

def get_random_matrix(n: int, k: int):
    if n > k:
        matrix = np.identity(n, dtype=int)
        matrix = np.array(matrix)

        randon_number_generator = default_rng()
        idxs: list = []
        for i in range(n):
            for j in range(n):
                while True:
                    idxs = randon_number_generator.choice(n, size=k, replace=False)
                    if i not in idxs:
                        break
            for idx in idxs:
                matrix[i, idx] = 1

        while not check_matrix(matrix, k):
            missing_ones = []
            excess_ones = []
            for j in range(n):
                if matrix[:, j].sum() < k + 1:
                    missing_ones.append(j)
                elif matrix[:, j].sum() > k + 1:
                    excess_ones.append(j)
            excess_col = randon_number_generator.choice(excess_ones, size=1)
            missing_col = randon_number_generator.choice(missing_ones, size=1)
            while True:
                row = np.random.randint(0, n)
                if (
                    matrix[row, missing_col] == 0
                    and matrix[row, excess_col] == 1
                    and row != excess_col
                    and row != missing_col
                ):
                    matrix[row, missing_col] = 1
                    matrix[row, excess_col] = 0
                    break
        return np.array(matrix)
    else:
        raise ValueError("Check values for parameters N and K")


def check_matrix(var: np.ndarray, k: int):
    if type(var) is np.ndarray and type(k) is int:
        if (
            var.shape[0] == var.shape[1]
            and all(var[i, :].sum() == k + 1 for i in range(var.shape[0]))
            and all(var[:, j].sum() == k + 1 for j in range(var.shape[0]))
        ):
            return True
        else:
            return False
    else:
        raise TypeError
