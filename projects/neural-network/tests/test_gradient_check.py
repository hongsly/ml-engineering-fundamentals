from typing import Callable

import numpy as np
from nn.utils import numerical_gradient
from nn.activations import ReLU, Sigmoid, Softmax
from nn.layers import Linear
from nn.module import Module
from nn.losses import CrossEntropyLoss, SoftmaxCrossEntropyLoss


def test_linear_weights():
    linear = Linear(input_dim=3, output_dim=2)
    x = np.array([[1.0, -2.0, 3.0]])
    grad_out = np.array([[0.5, -0.47]])

    _assert_grad(
        linear,
        x=x,
        grad_out=grad_out,
        get_analytic_grad=lambda module, grad_in: module.gradients["W"],
        parameter=linear.parameters["W"],
    )


def test_linear_bias():
    linear = Linear(input_dim=3, output_dim=2)
    x = np.array([[1.0, -2.0, 3.0]])
    grad_out = np.array([[0.5, -0.47]])

    _assert_grad(
        linear,
        x=x,
        grad_out=grad_out,
        get_analytic_grad=lambda module, grad_in: module.gradients["b"],
        parameter=linear.parameters["b"],
    )


def test_linear_input():
    linear = Linear(input_dim=3, output_dim=2)
    x = np.array([[1.0, -2.0, 3.0]])
    grad_out = np.array([[0.5, -0.47]])
    _assert_grad(linear, x=x, grad_out=grad_out)


def test_relu_backward():
    relu = ReLU()
    _assert_grad(relu)


def test_sigmoid_backward():
    sigmoid = Sigmoid()
    _assert_grad(sigmoid)


def test_softmax_backward():
    softmax = Softmax()
    _assert_grad(softmax)


def test_cross_entropy_loss_backward():
    cross_entropy_loss = CrossEntropyLoss()
    _assert_grad(cross_entropy_loss, x=np.array([[.6, .3, .5]]), y_true=np.array([[1.0, 0.0, 0.0]]), grad_out=1.0)


def test_softmax_cross_entropy_loss_backward():
    softmax_cross_entropy_loss = SoftmaxCrossEntropyLoss()
    _assert_grad(softmax_cross_entropy_loss, y_true=np.array([[1.0, 0.0, 0.0]]), grad_out=1.0)


def _assert_grad(
    module: Module,
    x: np.ndarray = np.array([[1.0, -2.0, 3.0]]),
    grad_out: np.ndarray = np.array([[0.5, -0.47, 0.3]]),
    get_analytic_grad: Callable[
        [Module, np.ndarray], np.ndarray
    ] = lambda module, grad_in: grad_in,
    parameter: np.ndarray = None,
    y_true: np.ndarray = None
) -> bool:
    """Assert analytic gradient matches numerical gradient. get_analytic_grad: (module, backward_output) -> analytic_grad"""
    if y_true is not None:
        module.forward(x, y_true)
    else:
        module.forward(x)  # needed to set self.input

    grad_in = module.backward(grad_out)
    grad_analytic = get_analytic_grad(module, grad_in)

    def loss_fn():
        if y_true is not None:
            y = module.forward(x, y_true)
        else:
            y = module.forward(x)
        print("x:", x, "y_true:", y_true, "y:", y)
        loss = np.sum(y * grad_out)  # artificial loss (like adding a linear layer)
        return loss

    grad_numerical = numerical_gradient(
        loss_fn, parameter if parameter is not None else x
    )

    assert np.allclose(grad_numerical, grad_analytic, atol=1e-7)


def test_full_network():
    linear1 = Linear(input_dim=3, output_dim=2)
    relu = ReLU()
    linear2 = Linear(input_dim=2, output_dim=3)
    softmax = Softmax()
    cross_entropy_loss = CrossEntropyLoss()

    x = np.array([[1.0, -2.0, 3.0]])
    y_true = np.array([[1.0, 0.0, 0.0]])
    y = softmax(linear2(relu(linear1(x))))
    loss = cross_entropy_loss(y, y_true)
    grad_in = linear1.backward(
        relu.backward(linear2.backward(softmax.backward(cross_entropy_loss.backward())))
    )

    def loss_fn():
        y_pred = softmax(linear2(relu(linear1(x))))
        return cross_entropy_loss(y_pred, y_true)

    grad_numerical_w1 = numerical_gradient(loss_fn, linear1.parameters["W"])
    grad_analytic_w1 = linear1.gradients["W"]
    assert np.allclose(grad_numerical_w1, grad_analytic_w1, atol=1e-7)

    grad_numerical_w2 = numerical_gradient(loss_fn, linear2.parameters["W"])
    grad_analytic_w2 = linear2.gradients["W"]
    assert np.allclose(grad_numerical_w2, grad_analytic_w2, atol=1e-7)

    grad_numerical_b1 = numerical_gradient(loss_fn, linear1.parameters["b"])
    grad_analytic_b1 = linear1.gradients["b"]
    assert np.allclose(grad_numerical_b1, grad_analytic_b1, atol=1e-7)

    grad_numerical_b2 = numerical_gradient(loss_fn, linear2.parameters["b"])
    grad_analytic_b2 = linear2.gradients["b"]
    assert np.allclose(grad_numerical_b2, grad_analytic_b2, atol=1e-7)

    grad_numerical_x = numerical_gradient(loss_fn, x)
    assert np.allclose(grad_numerical_x, grad_in, atol=1e-7)
