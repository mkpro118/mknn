import numpy as np
try:
    from utils import export_function
except ImportError:
    from .utils import export_function


@export_function
def normalize(vector: np.ndarray, /) -> np.ndarray:
    _ = np.max(np.abs(vector))
    return vector / _


@export_function
def centerify(vector: np.ndarray, /):
    return vector - np.mean(vector)


@export_function
def categorize(vector, /, *, classes: int = None):
    y = []
    if not classes:
        classes = np.max(vector) + 1
    for scalar in vector:
        _ = [0] * classes
        _[scalar] = 1
        y.append(_)
    return np.reshape(y, (len(vector), classes, 1))


@export_function
def train_test_split(vector1: np.ndarray,
                     vector2: np.ndarray,
                     /, *,
                     ratio: float = 0.8) -> tuple:
    if not (len(vector1) == len(vector2)):
        raise ValueError(f'Vector 1 and Vector 2 are not of '
                         f'the same dimension along the first axis.\n'
                         f'{vector1.shape = }\n'
                         f'{vector2.shape = }\n')
    _ = np.random.permutation(len(vector1))
    _X = vector1[_]
    _y = vector2[_]
    val = int(len(_X) * ratio)
    return _X[:val], _X[val:], _y[:val], _y[val:]
