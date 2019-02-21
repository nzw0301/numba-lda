import numpy as np


class Dirichlet(object):
    min_alpha = 10e-300

    def __init__(self, topic_size=10, coefficient=0.1):
        self.K = topic_size
        self.alpha = np.ones(self.K) * coefficient
        self.sum_alpha = np.sum(self.alpha)

    def get_alphas(self):
        return self.alpha

    def get_alpha(self, dim: int):
        return self.alpha[dim]

    def get_sum_alpha(self):
        return self.sum_alpha

    def sample(self, k: int, rnd: np.random.RandomState):
        return rnd.dirichlet(self.get_alphas(), k)
