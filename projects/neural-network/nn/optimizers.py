import numpy as np

from nn.module import Module


class Optimizer:

    def __init__(self, layers: list[Module]):
        self.layers = layers

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(
        self,
        layers: list[Module],
        learning_rate: float,
    ):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            for key in layer.parameters:
                layer.parameters[key] -= self.learning_rate * layer.gradients[key]

    def zero_grad(self):
        for layer in self.layers:
            for gradient in layer.gradients.values():
                gradient[:] = 0


class Adam(Optimizer):

    def __init__(
        self,
        layers: list[Module],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.optimizer_states = []
        for layer in layers:
            m = {}
            v = {}
            for key, parameter in layer.parameters.items():
                m[key] = np.zeros_like(parameter)
                v[key] = np.zeros_like(parameter)
            self.optimizer_states.append((m, v))

    def step(self):
        self.t += 1

        for layer, optimizer_state in zip(self.layers, self.optimizer_states):
            m, v = optimizer_state
            for key, parameter in layer.parameters.items():
                m[key] = self.beta1 * m[key] + (1 - self.beta1) * layer.gradients[key]
                v[key] = (
                    self.beta2 * v[key] + (1 - self.beta2) * layer.gradients[key] ** 2
                )
                m_hat = m[key] / (1 - self.beta1**self.t)
                v_hat = v[key] / (1 - self.beta2**self.t)

                parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for layer in self.layers:
            for gradient in layer.gradients.values():
                gradient[:] = 0
