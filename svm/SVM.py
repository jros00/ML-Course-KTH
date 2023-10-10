import numpy as np
import math
from scipy.optimize import minimize
from generate_training_samples import generate_training_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        self.feature_vectors: np.ndarray[np.ndarray] = feature_vectors
        self.label_vector: np.ndarray = label_vector
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


    def rbf_kernel(self, x1, x2, sigma=2.0):
        """
        ### Compute the RBF kernel (Gaussian kernel) between two data points.
        Uses the euclidian distance between two datapoints. Often very good boundaries.
        #### Parameters:
        - x1, x2: Numpy Arrays representing data points.
        - gamma: Parameter controlling the kernel's width and the smothness of the boundary (default=1.0)
        #### Returns:
        - The scalar product-like similarity measure.
        """
        return math.exp(-math.pow(np.linalg.norm(np.subtract(x1, x2)), 2)/(2 * math.pow(sigma, 2)))
    

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
            self.find_support_vectors()
            self.calculate_bias()
        else:
            raise ValueError("Optimization failed.")
        

    def find_support_vectors(self):
        """ ### Extract support-vectors """
        limit = 0.00001
        self.non_zeroes = [
            {
                'feature': self.feature_vectors[i],
                'target': self.label_vector[i],
                'alpha': self.alpha[i]
            }
        for i in range(len(self.alpha)) if self.alpha[i] >= limit]


    def calculate_bias(self) -> None:
        """
        ### Calculate the threshold value 'b' using the support vectors.
        #### The bias term is calculated using the support vectors like so:
        - b = ∑ alpha[i] * target[i] * kernel_matrix[[support_vector[i], x[i]]] - target[support_vector]
        """

        # Initialize b to the target value of the first support vector.
        # Use the first support vector as a reference point, the choice of support vector
        # doesn't affect the final decision boundary, as long as it's a support vector.
        b = -self.non_zeroes[0]['target']
        
        # loop through the support vectors
        for sv in self.non_zeroes:
            alpha_i = sv['alpha']
            feature_i = sv['feature']
            target_i = sv['target']
            
            # Calculate the kernel value between the current support vector and the first support vector
            first_sv = self.non_zeroes[0]['feature']
            if self.kernel == 'linear':
                kernel_value = self.linear_kernel(x1=feature_i, x2=first_sv)
            elif self.kernel == 'polynomial':
                kernel_value = self.polynomial_kernel(x1=feature_i, x2=first_sv)
            elif self.kernel == 'rbf':
                kernel_value = self.rbf_kernel(x1=feature_i, x2=first_sv)

            b += alpha_i * target_i * kernel_value
            
        self.bias = b
    

    def indicator(self, point: np.ndarray):
        """
        ### Use this function classify a new point.
        #### Parameters:
        - point: NumPy feature array [f1, f2]
        #### Returns:
        - 1 or 1 (positive / negative class)
        """
        # Initialize the sum to 0
        totsum = 0

        # Loop through the support vectors
        for sv in self.non_zeroes:
            alpha_i, feature_i, target_i = sv['alpha'], sv['feature'], sv['target']

            # Calculate the kernel value between the current input (x, y) and the support vector feature
            if self.kernel == 'linear':
                kernel_value = self.linear_kernel(x1=point, x2=feature_i)
            elif self.kernel == 'polynomial':
                kernel_value = self.polynomial_kernel(x1=point, x2=feature_i)
            elif self.kernel == 'rbf':
                kernel_value = self.rbf_kernel(x1=point, x2=feature_i)

            # Accumulate the weighted sum using alpha, target, and the kernel value
            totsum += alpha_i * target_i * kernel_value

        # Subtract the bias term to get the final decision value
        decision_value = totsum - self.bias

        # The indicator function output is 1 if the decision value is greater than or equal to 0, else -1
        if decision_value >= 0:
            return 1
        else:
            return -1
        

    def plot_decision_boundary(self):
        """
        Plot the decision boundary for the trained SVM model.
        """
        # Define the range of x and y values for the decision boundary
        x_min, x_max = self.feature_vectors[:, 0].min() - 1, self.feature_vectors[:, 0].max() + 1
        y_min, y_max = self.feature_vectors[:, 1].min() - 1, self.feature_vectors[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        # Create a grid of points to evaluate the indicator function
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.indicator(point) for point in grid])

        # Reshape the results to match the grid
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary using contour lines
        plt.contourf(xx, yy, Z, levels=[-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), alpha=0.5, linestyles='solid')

        # Classify and mark points as Class A (outside) and Class B (inside)
        for i in range(len(self.feature_vectors)):
            point = self.feature_vectors[i]
            if self.indicator(point) > 0:
                plt.scatter(point[0], point[1], c='blue', marker='o', label='Class A')
            else:
                plt.scatter(point[0], point[1], c='red', marker='o', label='Class B')

        plt.title(f'SVM Decision Boundary - {self.kernel.capitalize()} Kernel')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Create the legend with two labels (Class A and Class B)
        plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Classified as Class A', markerfacecolor='blue', markersize=10),
                            Line2D([0], [0], marker='o', color='w', label='Classified as Class B', markerfacecolor='red', markersize=10)],
                loc='upper right')
        plt.axis('equal') # force same scale and axises
        plt.savefig(f'svm/figures/svm-boundary-{self.kernel}-kernel.png') # save the copy
        plt.show()

    def plot_boundary(self):

        plt.figure(figsize=(10, 6))
        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        grid = np.array([[self.indicator([x, y]) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
        plt.show()



if __name__ == '__main__':
    feature_vectors, label_vector = generate_training_data(plot_data=False) # numpy arrays: [[1.3, 2.4], ...], [1, -1, ...]
    svm = SupportVectorMachine(
            feature_vectors=feature_vectors, 
            label_vector=label_vector, 
            kernel='linear',
            C=10
        )
    svm.train()
    svm.plot_decision_boundary()

