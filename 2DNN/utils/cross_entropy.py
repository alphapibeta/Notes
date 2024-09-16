import numpy as np

MAX_ENTROPY = 1

def cross_entropy(predictions=None, ground_truth=None):
    if predictions is None or ground_truth is None:
        raise Exception("Error! Both predictions and ground truth must be float32 arrays")

    p = np.array(predictions).copy()
    y = np.array(ground_truth).copy()

    if p.shape != y.shape:
        raise Exception("Error! Both predictions and ground_truth must have same shape.")

    if len(p.shape) != 2:
        raise Exception("Error! Both predictions and ground_truth must be 2D arrays.")

    total_entropy = 0

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if y[i, j] == 1:
                total_entropy += min(np.abs(np.nan_to_num(np.log(p[i, j]))), MAX_ENTROPY)
            else:
                total_entropy += min(np.abs(np.nan_to_num(np.log(1 - p[i, j]))), MAX_ENTROPY)

    return total_entropy / p.size