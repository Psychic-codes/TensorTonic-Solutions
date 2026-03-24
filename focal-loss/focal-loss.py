import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.array(p)
    y = np.array(y)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    loss = -((1 - p) ** gamma) * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)
    return np.mean(loss)
    pass