import numpy as np


def linear_kernel(**kwargs):
    def f(x, y):
        return np.inner(x, y)
    return f

def poly_kernel(power, coef, **kwargs):
    def f(x, y):
        return (np.inner(x, y) + coef)**power
    return f

def rbf_kernel(gamma, **kwargs):
    def f(x, y):
        dist = np.linalg.norm(x-y)**2
        return np.exp(-gamma*dist)
    return f

kernels = {'poly':poly_kernel, 'linear':linear_kernel, 'rbf':rbf_kernel}
