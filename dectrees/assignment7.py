# Import necessary modules
import monkdata as m
import dtree as dtree
import random
import numpy as np
import matplotlib.pyplot as plt

def partition(data, fraction):
    """ Function to partition dataset into training and validation sets """
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune_tree(tree, val_data):
    """ Function to prune the decision tree and return the pruned version with the best accuracy """
    best_tree = tree
    best_accuracy = dtree.check(best_tree, val_data)
    
    # Loop over all pruned versions of the tree
    for pruned_tree in dtree.allPruned(tree):
        acc = dtree.check(pruned_tree, val_data)
        if acc >= best_accuracy:
            best_tree, best_accuracy = pruned_tree, acc

    return best_tree


def compute_pruned_validation_statistics(datasets: dict, fractions: list, n: int):
    """ Function to compute statistics after pruning the trees """
    stats = dict()
    attributes = m.attributes 

    # Loop through each dataset (monk1, monk3)
    for name, dataset in datasets.items():
        
        # Loop through each fraction to partition data
        for fraction in fractions:
            accuracies = []

            # Run n times for each fraction
            for _ in range(n):
                train, val = partition(data=dataset, fraction=fraction)
                
                # Build initial decision tree
                initial_tree = dtree.buildTree(train, attributes)
                
                # Prune the tree
                pruned_tree = prune_tree(initial_tree, val)
                
                # Compute accuracy
                acc = dtree.check(pruned_tree, val)
                accuracies.append(acc)

            mean_acc = np.mean(accuracies)
            std_dev = np.std(accuracies)  # Standard deviation for the measure of spread

            if name not in stats:
                stats[name] = {}

            # Store statistics
            stats[name][fraction] = {'mean': mean_acc, 'std_dev': std_dev}

    return stats


def main():
    # monk datasets
    datasets = {
        'monk1': m.monk1,
        'monk3': m.monk3
    }

    # Fractions to test
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Number of iterations for each fraction
    n = 500

    stats = compute_pruned_validation_statistics(datasets, fractions, n)

    # Initialize the plot with better sizing and a clearer style
    plt.figure(figsize=(12, 8), facecolor='white')

    # Plotting
    for name in stats.keys():
        means = [stats[name][fraction]['mean'] for fraction in fractions]
        std_devs = [stats[name][fraction]['std_dev'] for fraction in fractions]
        
        # Using stylish markers, line styles, and setting capsize for error bars
        plt.errorbar(fractions, means, yerr=std_devs, label=f"{name} Data", marker='o', linestyle='-', capsize=5)

    # Customize x and y labels with font size
    plt.xlabel('Fraction of Data', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)

    # Add a descriptive title with larger font size
    plt.title('Pruned Tree Validation', fontsize=16)

    # Add grid for better readability of plotted points
    plt.grid(True, linestyle='--', linewidth=0.5, color='grey')

    # Add a legend with a specified location
    plt.legend(loc='upper left', fontsize='large')

    # Show plot with cleaner UI
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
