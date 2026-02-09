import numpy as np


class Module:
    """
    Base class for all modules in the neural network.
    Mimic PyTorch's nn.Module pattern.
    """

    def __init__(self) -> None:
        self.parameters: dict[str, np.ndarray] = {}
        self.gradients: dict[str, np.ndarray] = {}

    def forward(self, *args) -> np.ndarray | float:
        """Compute the forward pass of the module. Must be implemented by the subclass."""
        raise NotImplementedError

    def backward(self, grad: np.ndarray | float = 1.0) -> np.ndarray:
        """Compute the backward pass of the module. Must be implemented by the subclass.

        Calculates and stores the gradients w.r.t. the parameters of the module,
        returns the gradient w.r.t. the module's input for backpropagation.
        """
        raise NotImplementedError

    def __call__(self, *args) -> np.ndarray | float:
        return self.forward(*args)
