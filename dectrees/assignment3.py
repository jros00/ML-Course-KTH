from monkdata import monk1, monk2, monk3, attributes
from dtree import averageGain

"""
Information Gain measures how much uncertainty is reduced in predicting a target variable, 
often used in decision tree algorithms. It's calculated as:

    - Information Gain = Entropy(Original) - Sum(Weighted Entropy of Subsets)

Higher value of information gain is better
"""

datasets = [monk1, monk2, monk3]
average_per_attribute = {
    'A1': [],
    'A2': [],
    'A3': [],
    'A4': [],
    'A5': [],
    'A6': []
}
for i in range(len(datasets)):
    dataset = datasets[i]
    for j in range(len(attributes)):
        attribute = attributes[j]
        information_gain = round(averageGain(dataset=dataset, attribute=attribute), 5)
        average_per_attribute[f'A{j+1}'].append(information_gain)
        print(f"Information Gain (Monk{i+1}, A{j+1}): {information_gain}")

for key in average_per_attribute:
    average_per_attribute[key] = round((sum(average_per_attribute[key])/len(average_per_attribute[key])), 5)
print(average_per_attribute)

"""
OUTPUT

    Information Gain (Monk1, A1): 0.07527
    Information Gain (Monk1, A2): 0.00584
    Information Gain (Monk1, A3): 0.00471
    Information Gain (Monk1, A4): 0.02631
    Information Gain (Monk1, A5): 0.28703
    Information Gain (Monk1, A6): 0.00076
    Information Gain (Monk2, A1): 0.00376
    Information Gain (Monk2, A2): 0.00246
    Information Gain (Monk2, A3): 0.00106
    Information Gain (Monk2, A4): 0.01566
    Information Gain (Monk2, A5): 0.01728
    Information Gain (Monk2, A6): 0.00625
    Information Gain (Monk3, A1): 0.00712
    Information Gain (Monk3, A2): 0.29374
    Information Gain (Monk3, A3): 0.00083
    Information Gain (Monk3, A4): 0.00289
    Information Gain (Monk3, A5): 0.25591
    Information Gain (Monk3, A6): 0.00708


    {'A1': 0.02872, 'A2': 0.10068, 'A3': 0.0022, 'A4': 0.01495, 'A5': 0.18674, 'A6': 0.0047}

END OF OUTPUT

EXPLANATION

    A5 has the highest average Information Gain, as such - use the A5 feauture to split the data.
    
    When building a decision tree, the aim is to partition the data in a way that minimizes 
    uncertainty or randomness about the target variable. 
    Each attribute (also known as a feature) in the dataset provides some amount of 
    information that can help in making a decision. 
    Information Gain is the metric used to quantify how much an attribute reduces this uncertainty.

    When you "pick the attribute that yields the highest Information Gain as the node to split the data," 
    it means that you choose the attribute that most effectively separates the data into 
    subsets where the target variable (like "Yes" or "No" for playing tennis) is more consistently one value 
    or another.

    In simpler terms, this attribute does the best job of separating your data into groups that are 
    easier to make predictions about. 
    For example, if you're trying to predict whether someone will play 
    tennis and you find that weather has the highest Information Gain, 
    it means that knowing the weather condition significantly improves your 
    ability to predict whether or not someone will play tennis.

    By repeatedly doing this at each node—picking the next best attribute to split on - you 
    build a decision tree that aims to make the most accurate predictions with the 
    least amount of uncertainty.

"""