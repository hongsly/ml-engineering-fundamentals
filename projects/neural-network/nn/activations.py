import numpy as np
from nn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch_size, dim)
        self.input = x
        return np.maximum(0, x)  # (batch_size, dim)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch_size, dim)
        return grad * (self.input > 0)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x  # x: (batch_size, dim)
        return 1 / (1 + np.exp(-x)) # (batch_size, dim)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (np.exp(-self.input) / (1 + np.exp(-self.input)) ** 2)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch_size, dim)
        # numerically stable softmax by subtracting the max value
        max_value = np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x - max_value)
        # self.output: (batch_size, dim)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch_size, num_classes)
        # s: (batch_size, num_classes)
        # TODO(later): derive this formula
        s = self.output
        grad_sum = np.sum(grad * s, axis=1, keepdims=True)
        return s * (grad - grad_sum)