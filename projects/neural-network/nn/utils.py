import numpy as np


def numerical_gradient(loss_fn, param, epsilon=1e-5):
    grad = np.zeros_like(param)
    # (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        param[idx] = old_val + epsilon
        pos = loss_fn()
        param[idx] = old_val - epsilon
        neg = loss_fn()
        grad[idx] = (pos - neg) / (2 * epsilon)

        param[idx] = old_val
        it.iternext()

    return grad
