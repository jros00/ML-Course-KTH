import math
from monkdata import Sample

def entropy(dataset: list[Sample]):
    """Calculate the entropy of a dataset.
    Importance: Used to determine how to split the data at each node.
    """
    n = len(dataset)
    nPos = len([x for x in dataset if x.positive])
    nNeg = n - nPos
    if nPos == 0 or nNeg == 0:
        return 0.0  # Entropy is zero when all samples are of the same class
    return -float(nPos)/n * log2(float(nPos)/n) + -float(nNeg)/n * log2(float(nNeg)/n)

def averageGain(dataset, attribute):
    """Calculate the expected information gain for a given attribute.
    Importance: Determines which attribute should be used for the next split.
    """
    weighted = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        weighted += entropy(subset) * len(subset)
    return entropy(dataset) - weighted / len(dataset)

def log2(x):
    """Compute logarithm to the base 2.
    Importance: Utility function used in entropy calculation.
    """
    return math.log(x, 2)

def select(dataset, attribute, value):
    """Return subset of data samples where an attribute has a given value.
    Importance: Helps create subsets of data that form child nodes.
    """
    return [x for x in dataset if x.attribute[attribute] == value]

def bestAttribute(dataset, attributes):
    """Find the attribute with the highest expected information gain.
    Importance: Critical for building an efficient decision tree.
    """
    gains = [(averageGain(dataset, a), a) for a in attributes]
    return max(gains, key=lambda x: x[0])[1]

def allPositive(dataset):
    """Check if all samples in the dataset are positive.
    Importance: Helps in deciding when to stop splitting a node.
    """
    return all([x.positive for x in dataset])

def allNegative(dataset):
    """Check if all samples in the dataset are negative.
    Importance: Helps in deciding when to stop splitting a node.
    """
    return not any([x.positive for x in dataset])

def mostCommon(dataset):
    """Determine the majority class in the dataset.
    Importance: Used as the default class when no further splitting is possible.
    """
    pCount = len([x for x in dataset if x.positive])
    nCount = len([x for x in dataset if not x.positive])
    return pCount > nCount

class TreeNode:
    """Representation of a decision tree node.
    Importance: Serves as the building block for the decision tree.
    """
    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'

class TreeLeaf:
    """Representation of a leaf node in the decision tree.
    Importance: Holds the final classification value for a subset of data.
    """
    def __init__(self, cvalue):
        self.cvalue = cvalue

    def __repr__(self):
        return '+' if self.cvalue else '-'

def buildTree(dataset, attributes, maxdepth=1000000):
    """Recursively build a decision tree.
    Importance: This is where the actual tree is built using a recursive algorithm.
    """
    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)
        if allPositive(dataset):
            return TreeLeaf(True)
        if allNegative(dataset):
            return TreeLeaf(False)
        return buildTree(dataset, attributes, maxdepth-1)

    default = mostCommon(dataset)
    if maxdepth < 1:
        return TreeLeaf(default)
    a = bestAttribute(dataset, attributes)
    attributesLeft = [x for x in attributes if x != a]
    branches = [(v, buildBranch(select(dataset, a, v), default, attributesLeft)) for v in a.values]
    return TreeNode(a, dict(branches), default)

def classify(tree, sample):
    """Classify a sample using the given decision tree.
    Importance: Makes predictions on new or test data.
    """
    if isinstance(tree, TreeLeaf):
        return tree.cvalue
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)

def check(tree, testdata):
    """Measure fraction of correctly classified samples.
    Importance: Used to evaluate the accuracy of the decision tree.
    """
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)

def allPruned(tree):
    """Generate all pruned versions of the given tree.
    Importance: Used for post-pruning to optimize the tree.
    """
    if isinstance(tree, TreeLeaf):
        return ()
    alternatives = (TreeLeaf(tree.default),)
    for v in tree.branches:
        for r in allPruned(tree.branches[v]):
            b = tree.branches.copy()
            b[v] = r
            alternatives += (TreeNode(tree.attribute, b, tree.default),)
    return alternatives
