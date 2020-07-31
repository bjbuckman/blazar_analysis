import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import poisson

source_rate = 1.
background_rate = np.logspace(-2, 2, base=10., num=20)
EXPO = 1.0

A = 3.
mu = (source_rate + background_rate)*EXPO
var = np.sqrt((source_rate + background_rate)*EXPO)

prob = 1.- poisson.cdf(np.floor(mu+A*var), mu)

print(prob)