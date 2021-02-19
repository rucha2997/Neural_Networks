# Gurav, Rucha
# 1001-773-732
# 2020-09-24
# Assignment-01-02

import numpy as np
import pytest
from single_layer_nn import SingleLayerNN


def test_set_and_get_weights():
    input_dimensions = 4
    number_of_nodes = 9
    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    weights=model.get_weights()
    assert weights.ndim == 2 and \
           weights.shape[0] == number_of_nodes and \
           weights.shape[1] == (input_dimensions + 1)
    model.set_weights(np.ones((number_of_nodes, input_dimensions + 1)))
    weights = model.get_weights()
    assert weights.ndim == 2 and \
           weights.shape[0] == number_of_nodes and \
           weights.shape[1] == (input_dimensions + 1)
    assert np.array_equal(model.get_weights(), np.ones((number_of_nodes, input_dimensions + 1)))


def test_weight_initialization():
    input_dimensions = 2
    number_of_nodes = 5
    model = SingleLayerNN(input_dimensions=2, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=1)
    assert model.weights.ndim == 2 and model.weights.shape[0] == number_of_nodes and model.weights.shape[
        1] == input_dimensions + 1
    weights = np.array([[1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862, 0.86540763, -2.3015387],
                        [1.74481176, -0.7612069, 0.3190391],
                        [-0.24937038, 1.46210794, -2.06014071],
                        [-0.3224172, -0.38405435, 1.13376944]])
    np.testing.assert_allclose(model.get_weights(), weights, rtol=1e-3, atol=1e-3)


def test_predict():
    input_dimensions = 2
    number_of_nodes = 2
    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    Y_hat = model.predict(X_train)
    assert np.array_equal(Y_hat, np.array([[1, 1, 1, 1], [1, 0, 1, 1]]))


def test_train_and_error_calculation():
    input_dimensions = 6
    number_of_nodes = 2
    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    X_train=np.random.randn(input_dimensions,5)
    X_train = np.array([[ 0.60302512,-0.94856749,-0.88904878,-0.02272178, 0.90226341],
 [ 0.23682103,-0.29573089,-0.2328635 , 0.85537468, 0.19151025],
 [-0.35294026,-0.99263065, 0.41645806,-0.22561292, 2.72811021],
 [ 0.06180665, 1.0643302 , 0.49739215,-1.81960612, 0.50104263],
 [-0.4875518 ,-0.98996947, 2.38729703, 0.95753127,-0.20929545],
 [-1.20261055,-1.84727613,-1.13450254,-1.57499346, 0.4382195 ]])
    Y_train = np.array([[0, 1, 1, 1,0], [0, 0, 1, 0,1]])
    model.initialize_weights(seed=2)
    error = []
    for k in range(20):
        error.append(model.calculate_percent_error(X_train, Y_train))
        model.train(X_train, Y_train, num_epochs=1, alpha=0.025)
    error.append(model.calculate_percent_error(X_train, Y_train))
    np.testing.assert_allclose(error, [80.0, 80.0, 80.0, 60.0, 60.0,
        60.0, 60.0, 60.0, 60.0, 60.0, 40.0, 20.0, 20.0, 20.0, 20.0,
        20.0, 20.0, 20.0, 20.0, 0.0, 0.0], rtol=1e-3, atol=1e-3)