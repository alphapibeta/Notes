import numpy as np

"""
    The function `condition_data` normalizes the input data by subtracting the mean and dividing by the
    standard deviation along each column, and returns the normalized data along with the means and
    standard deviations used for normalization.
    
    :param data: The `data` parameter is the input data that you want to condition. It is typically a
    numpy array where each row represents a sample and each column represents a feature
    :param means: The `means` parameter in the `condition_data` function represents the mean values of
    the data along each column (axis=0). If the `means` parameter is not provided when calling the
    function, it calculates the mean values of the data array along the columns using `np.mean(data,
    axis
    :param stds: The `stds` parameter in the `condition_data` function represents the standard
    deviations of the input data along each column (axis=0). It is used to normalize the data by
    subtracting the mean and dividing by the standard deviation to make the data have a mean of 0 and a
    standard
    :return: The function `condition_data` returns three values: 
    1. `conditioned_data`: the input data after being standardized (mean-centered and scaled by standard
    deviation)
    2. `means`: the means of the input data along each column
    3. `stds`: the standard deviations of the input data along each column
"""

def condition_data(data, means=None, stds=None):
    if means is None:
        means = np.mean(data, axis=0)
    if stds is None:
        stds = np.std(data, axis=0)
    conditioned_data = data.copy()
    conditioned_data -= means
    conditioned_data /= stds
    return conditioned_data, means, stds