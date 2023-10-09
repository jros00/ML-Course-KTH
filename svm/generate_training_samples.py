import numpy as np
import random
import matplotlib.pyplot as plt

def generate_training_data(plot_data: bool = False):

    """
    ## Inputs:
    - plot_data: bool. If true, plots the data.
    ## Returns:
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

    if plot_data:
        plot(class_a=class_a, class_b=class_b)

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


def plot(class_a: np.ndarray, class_b: np.ndarray):

    plt.plot(
        [p[0] for p in class_a],
        [p[1] for p in class_a],
        'b.',
        label='Class A'
    )

    plt.plot(
        [p[0] for p in class_b],
        [p[1] for p in class_b],
        'r.',
        label='Class B'
    )

    plt.legend()
    plt.axis('equal') # force same scale and axises
    plt.savefig('svm/figures/svmplot.png') # save the copy
    plt.show()



if __name__ == '__main__':
    inputs, targets = generate_training_data(plot_data=True)
    print(inputs, targets)