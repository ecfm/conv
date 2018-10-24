from __future__ import division
import math


class SGDMomentum(object):
    def __init__(self, model, mu, epsilon, gamma, power, weight_decay):
        self.mu = mu
        self.epsilon = epsilon
        self.gamma = gamma
        self.power = power
        self.weight_decay = weight_decay
        self.param = []
        self.theta = []
        for layer in model._network:
            if hasattr(layer, "params"):
                for p_key in layer.params:
                    self.param += [layer.params[p_key]]
                    self.theta += [None]
        self.alpha = None

    def zero_grad(self):
        for p in self.param:
            p.grad = None

    def update_lr(self, itr):
        pass
        return self.alpha

    def step(self):
        pass


class SGD(object):
    """
    A basic SGD
    """
    def __init__(self, parameters, lr=0.1):
        self.lr = lr
        self.parameters = parameters

    def zero_grad(self):
        for group in self.parameters:
            for param in self.parameters[group]:
                self.parameters[group][param].grad = None

    def step(self):
        for group in self.parameters:
            for param in self.parameters[group]:
                self.parameters[group][param].value -= self.lr * self.parameters[group][param].grad

    def update_lr(self, itr):
        pass
