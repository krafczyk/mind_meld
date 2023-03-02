import numpy as np
import numba
from numba.compiler import compile_isolated

def transform_1(x):
    return x**2

def transform_2(x):
    return x+np.eye(x.shape[0], x.shape[1]);

if __name__ == "__main__":
    array_shape = (5, 5)
    num_examples = 10000
    in_data = np.random.random((num_examples,)+array_shape)
    print(in_data.shape)

    for el in in_data:
        print(el.shape)
        print(el)
        print(transform_1(el))
        print(transform_2(transform_1(el)))
        break
