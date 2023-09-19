from monkdata import monk1, monk2, monk3
from dtree import entropy

ent1 = entropy(dataset=monk1)
ent2 = entropy(dataset=monk2)
ent3 = entropy(dataset=monk3)

print(
    f"\nEntropy - monk1: {ent1}\nEntropy - monk2: {ent2}\nEntropy - monk3 {ent3}\n"
)

"""
Output:

    Entropy - monk1: 1.0
    Entropy - monk2: 0.957117428264771
    Entropy - monk3 0.9998061328047111
"""
