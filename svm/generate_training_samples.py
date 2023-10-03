import numpy as np
import random

def generate_training_data():

    """
    ### Returns:
    - inputs: A nested numpy array with input points [[1.3, 0.4], ...]
    - targets: A numpy array with the input points classification [-1, 1, ..., -1] where -1 is the negative class and 1 the positive.
    """
    class_a = np.concatenate(
        (
            np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
            np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
        )
    )

    class_b = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

    inputs = np.concatenate((class_a, class_b))
    targets = np.concatenate(
        (
            np.ones(class_a.shape[0]),
            -np.ones(class_b.shape[0])
        )
    )

    N = inputs.shape[0] # number of rows

    # randomly reorder sample
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets


if __name__ == '__main__':
    inputs, targets = generate_training_data()
    print(inputs, targets)