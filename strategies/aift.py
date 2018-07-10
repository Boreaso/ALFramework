import numpy as np


def _majority(probs, num, log=False):
    probs = np.sort(probs)
    mean = np.mean(probs)
    if mean > 0.5:
        probs = probs[-num:]
        if log:
            print(" select top %d probs." % num)
    else:
        probs = probs[:num]
        if log:
            print(" select botom %d probs." % num)
    return probs


def _compute_entropy(probs, i):
    assert isinstance(probs, np.ndarray) and i < len(probs)
    eps = np.spacing(1)
    if not isinstance(probs[i], np.ndarray) or len(probs[i]) == 1:
        entropy = -(probs[i] * np.log(probs[i] + eps) + (1 - probs[i]) * np.log(1 - probs[i] + eps))
    else:
        entropy = np.sum(-(probs[i] * np.log(probs[i] + eps)))
    return entropy


def _compute_diversity(probs, i, j):
    assert isinstance(probs, np.ndarray)
    assert j < len(probs) and j < len(probs)
    eps = np.spacing(1)
    if not isinstance(probs[i], np.ndarray) or len(probs[i]) == 1:
        diversity = (np.array(probs[i] - probs[j])) * np.log((probs[i] + eps) / (probs[j] + eps)) + \
                    (np.array(probs[j]) - np.array(probs[i])) * np.log((1 - probs[i] + eps) / (1 - probs[j] + eps))
    else:
        diversity = np.sum((np.array(probs[i]) - np.array(probs[j])) * np.log((probs[i] + eps) / (probs[j] + eps)))
    return diversity


def compute_r_matrix(probs, lambda_1=1, lambda_2=0, alpha=1):
    """
    Compute R matrix proposed in paper: Fine-tuning Convolutional Neural Networks
    for Biomedical Image Analysis: Actively and Incrementally
    probs: CNN prediction probabilities
    (lamda_1 and lamda_2 are trade-offs between entropy and diversity)
    lamda_1: turn on/off entropy (1 or 0)
    lamda_2: turn on/off diversity (1 or 0)
    alpha: percentage of selection
    """
    num = round(len(probs) * alpha)
    probs = _majority(probs, num)

    r_matrix = np.zeros(shape=(num, num))
    for i in range(num):
        for j in range(num):
            if i == j:
                r_matrix[i][j] = lambda_1 * _compute_entropy(probs, i)
            else:
                r_matrix[i][j] = lambda_2 * _compute_diversity(probs, i, j)

    return np.round(np.sum(r_matrix), 2)


if __name__ == '__main__':
    _probs = [[0.1, 0.3, 0.6], [0.4, 0.5, 0.1], [0.5, 0.3, 0.2]]
    matrix_sum = compute_r_matrix(_probs)
    print(matrix_sum)
