import numpy as np

def condition_data(data, means=None, stds=None):
    if means is None:
        means = np.mean(data, axis=0)
    if stds is None:
        stds = np.std(data, axis=0)
    conditioned_data = data.copy()
    conditioned_data -= means
    conditioned_data /= stds
    return conditioned_data, means, stds