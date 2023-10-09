import numpy as np
from scipy.optimize import minimize
from generate_training_samples import generate_training_data

class SupportVectorMachine:
    """
    ### A class for a Support Vector Machine.
    #### Params:
    - feature_vectors: Nested NumPy Arrays with the features of a point.
    - label_vector: A NumPy Array of the labels of each point, in the same order.
    - kernel: The chosen kernel function. 'linear' | 'polynomial' | 'rbf'
    - C: the regularization parameter
    """
    def __init__(self, feature_vectors, label_vector, kernel='linear', C=1.0) -> None:
        self.feature_vectors = feature_vectors
        self.label_vector = label_vector
        self.kernel = kernel
        self.C = C
        self.alpha = self.initialize_alpha()
        self.kernel_matrix = self.create_kernel_matrix()
        self.non_zeroes = None

    def linear_kernel(self, x1, x2):
        """
        ### Compute the linear kernel between two data points. 
        #### Parameters:
        - x1, x2: Numpy Arrays representing data points.
        #### Returns:
        - Scalar product. This results in a linear separation.
        - The scalar product-like similarity measure.
        """
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=2):
        """
        ### Compute the polynomial kernel between two data points.
        Allows for curved decision boundaries.
        #### Parameters:
        - x1, x2: Numpy Arrays representing data points.
        - degree: Degree of the polynomial kernel (default=2). degree = 2 will give quadratic shapes.
        #### Returns:
        - The scalar product-like similarity measure.
        """
        return (np.dot(x1, x2) + 1) ** degree

    def rbf_kernel(self, x1, x2, gamma=1.0):
        """
        ### Compute the RBF kernel (Gaussian kernel) between two data points.
        Uses the euclidian distance between two datapoints. Often very good boundaries.
        #### Parameters:
        - x1, x2: Numpy Arrays representing data points.
        - gamma: Parameter controlling the kernel's width and the smothness of the boundary (default=1.0)
        #### Returns:
        - The scalar product-like similarity measure.
        """
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff, diff))
    
    def initialize_alpha(self) -> np.ndarray:
        """ ### Returns alpha, initialized to an array filled with zeroes. """
        return np.zeros(len(self.label_vector))
    
    def create_kernel_matrix(self) -> np.ndarray:
        """
        ### Compute the kernel matrix, K
        #### Returns: 
            - the kernel matrix (N x N)
        """
        n = len(self.alpha)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Use the chosen kernel function to compute K
                x1, x2 = self.feature_vectors[i], self.feature_vectors[j]
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(x1=x1, x2=x2)
                elif self.kernel == 'polynomial':
                    K[i, j] = self.polynomial_kernel(x1=x1, x2=x2)
                elif self.kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(x1=x1, x2=x2)
        return K 

    def objective(self, alpha: np.ndarray):
        """
        ### Compute the objective function for the dual SVM problem.
        #### Returns:
        - The value of the objective function.
        """

        # Compute the kernel matrix weighted by alpha and label_vector
        weighted_kernel_matrix = np.outer(alpha * self.label_vector, alpha * self.label_vector) * self.kernel_matrix

        # Calculate the objective value
        obj_value = np.sum(weighted_kernel_matrix) - np.sum(alpha)

        return obj_value

    def zerofun(self, alpha):
        """ 
        ### Returns the value of ∑ α[i]t[i]
        #### If 0, both conditions are fulfilled (i.e) ∑ α[i]t[i] = 0 and 0 <= α[i] <= C.
        #### If any α[i] violates the constraint (i.e., becomes negative or exceeds C), the dot product would not be zero.
        """
        return np.dot(alpha, self.label_vector) # should be 0.
    
    def train(self):

        """ 
        ### Sets 'self.alpha'
        #### self.alpha is a vector that represents the Lagrange multipliers associated with each data point in your training dataset.
        """

        alpha = self.alpha.copy()

        # Define the bounds for alpha (0 <= alpha <= C) for all alpha
        bounds = [(0, self.C) for _ in range(len(alpha))]

        # Define the equality constraint (∑(alpha * label_vector) = 0)
        constraint = {'type': 'eq', 'fun': self.zerofun}

        # Perform the optimization using the minimize function
        result = minimize(self.objective, alpha, bounds=bounds, constraints=constraint)

        # Check if the optimizer effectively found a solution
        if result.success:
            self.alpha = result.x
            print("Optimization successful.")
        else:
            raise ValueError("Optimization failed.")
        
        self.non_zeroes = [
            {
                'feature': self.feature_vectors[i],
                'target': self.label_vector[i],
                'alpha': self.alpha[i]
            }
        for i in range(len(self.alpha)) if self.alpha[i] <= 0.00001]

if __name__ == '__main__':
    feature_vectors, label_vector = generate_training_data(plot_data=False) # numpy arrays: [[1.3, 2.4], ...], [1, -1, ...]
    print(feature_vectors, label_vector)
    svm = SupportVectorMachine(
            feature_vectors=feature_vectors, 
            label_vector=label_vector, 
            kernel='linear',
            C=10
        )
    svm.train()
    print(svm.alpha)
    print(svm.non_zeroes)
