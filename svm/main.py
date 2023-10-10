from SVM import SupportVectorMachine
from generate_training_samples import generate_training_data

feature_vectors, label_vector = generate_training_data(plot_data=True)
for kernel in ['linear', 'polynomial', 'rbf']:
    svm = SupportVectorMachine(
            feature_vectors=feature_vectors, 
            label_vector=label_vector, 
            kernel=kernel,
            C=10
        )
    svm.train()
    svm.plot_decision_boundary()