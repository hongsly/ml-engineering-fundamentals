import numpy as np
from nn.module import Module


class SoftmaxCrossEntropyLoss(Module):
    # cross-entropy loss on logits output

    def __init__(self):
        super().__init__()

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        # logits: (batch_size, num_classes)
        # y_true: (batch_size, num_classes)
        epsilon = 1e-8
        self.y_true = y_true
        batch_size = logits.shape[0]
        max_logits = np.max(logits, axis=1, keepdims=True)
        adjusted_logits = logits - max_logits
        self.logits = adjusted_logits
        return -np.sum(y_true * (adjusted_logits - np.log(epsilon + np.sum(np.exp(adjusted_logits), axis=1, keepdims=True)))) / batch_size

    def backward(self, grad: float = 1.0) -> np.ndarray:
        batch_size = self.logits.shape[0]
        softmax = np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1, keepdims=True)
        return (softmax - self.y_true) * grad / batch_size


class CrossEntropyLoss(Module):
    # Cross-entropy loss on softmax output

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size, num_classes)
        epsilon = 1e-8
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = y_pred.shape[0]
        return -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size

    def backward(self, grad: float = 1.0) -> np.ndarray:
        # return shape: (batch_size, num_classes)
        epsilon = 1e-8
        batch_size = self.y_pred.shape[0]
        return -self.y_true / (self.y_pred + epsilon) * grad / batch_size


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # y_pred: (batch_size, 1)
        # y_true: (batch_size, 1)
        batch_size = y_pred.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true
        return np.sum((y_pred - y_true) ** 2) / batch_size

    def backward(self, grad: float = 1.0) -> np.ndarray:
        batch_size = self.y_pred.shape[0]
        # return shape: (batch_size, 1)
        return 2 * (self.y_pred - self.y_true) / batch_size * grad
