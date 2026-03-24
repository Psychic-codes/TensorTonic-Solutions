import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.array(y)
    
    # Get class counts
    values, counts = np.unique(y, return_counts=True)
    
    # Compute probabilities
    probs = counts / counts.sum()
    
    # Compute entropy (stable: ignore zero probabilities)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy
    pass