from keras.optimizers import SGD, Adam
from keras.layers import Dense, Activation, Input, Dropout, merge
from keras.models import Model, Sequential
import numpy as np


class GaussianDistribution:
    """
     Constructing a Gaussian distribution with functions to get  log likelihood,
     dimension, samples,energy and gradient
    """

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def __logLikelihood_gaussian(self, x):
        return(-0.5 * np.sum((x - self.mu) * (1.0 / self.var) * (x - self.mu)) -
               0.5 * np.log(np.prod(2*np.pi*self.var)))

    def sample(self, n):
        dim = self.mu.shape[0]
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = np.random.normal(
                self.mu[i], np.sqrt(self.var[i]), n)
        return(samples)

    def get_dim(self):
        return(self.mu.shape[0])

    def energy(self, x):
        return(-self.__logLikelihood_gaussian(x))

    def gradient(self, x):
        return(-(x - self.mu) / self.var)


def build(input_dim, hidden_dim, output_dim):
    """Builds a neural network for gradient approximation.

    The neural network consists of individual blocks for partial gradients.

    Args:
        input_dim: int for input dimension
        hidden_dim: list of int for multiple hidden layers
        output_dim: list of int for multiple output layers

    Returns:
        A complied Keras model.
    """
    x = Input(shape=(input_dim,))

    hidden = []
    for s in hidden_dim:
        hidden.append(Dense(s, activation='relu')(x))

    output = []
    for i, h in enumerate(hidden):
        output.append(
            Dense(output_dim[i], activation='linear')(h))

    if len(output) == 1:
        y = output[0]
    else:
        y = merge(output, mode='concat', concat_axis=1)

    model = Model(x, y)
    model.summary()
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return(model)
