import numpy as np


class Param(object):
    def __init__(self, name):
        self.name = name

    def sample(self):
        pass

    def mutate(self, value):
        pass


class CategoricalParam(Param):
    def __init__(self, name, categories, prior=None):
        super(CategoricalParam, self).__init__(name)
        self.categories = categories
        self.prior = prior
        if self.prior is not None:
            assert len(self.prior) == len(categories)
            assert np.all(np.array(self.prior) > 0)
            self.prior /= np.sum(1.0 * np.array(self.prior))

    def sample(self):
        return np.random.choice(self.categories, p=self.prior)

    def mutate(self, category):
        return np.random.choice([c for c in self.categories if c != category])


class ContinuousParam(Param):
    def __init__(self, name, mean, variance, min_limit=None, max_limit=None, is_int=False):
        super(ContinuousParam, self).__init__(name)
        self.mean = mean
        self.variance = variance
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.is_int = is_int

    def _correct(self, value):
        if self.max_limit is not None:
            value = min(value, self.max_limit)
        if self.min_limit is not None:
            value = max(value, self.min_limit)
        if self.is_int:
            value = int(value)
        return value

    def sample(self):
        value = np.random.normal(self.mean, self.variance)
        return self._correct(value)

    def mutate(self, value):
        sign = 2 * (np.random.rand() > 0.5) - 1
        magnitude = ((np.random.rand() > 0.5) + 1) * 0.25
        return self._correct((1.0 + sign * magnitude) * value)


class ConstantParam(Param):
    def __init__(self, name, value):
        super(ConstantParam, self).__init__(name)
        self.value = value

    def sample(self):
        return self.value

    def mutate(self, value):
        return value
