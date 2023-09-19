from monkdata import monk1, monk2, monk3, monk1test, monk2test, monk3test, attributes
import dtree


# t = dtree.buildTree(monk1, attributes)
# print(dtree.check(t, monk1test))

train = [monk1, monk2, monk3]
test = [monk1test, monk2test, monk3test]

for i in range(len(train)):
    train_dataset = train[i]
    test_dataset = test[i]
    tree = dtree.buildTree(train_dataset, attributes)
    test_acc = round(dtree.check(tree, test_dataset), 5)
    train_acc = round(dtree.check(tree, train_dataset), 5)
    print(f"\nTraining Dataset: Monk{i+1}.\n     - Test Accuray: {test_acc}.\n     - Training Accuracy: {train_acc}")


"""
Explanation:
    Our assumptions were correct.
    
    We predicted that Monk2 should be the hardest to predict, since it has the most simultaneous
    comparison between Attributes, which is something that typically
    requires more than a binary split.

    Also, it is logical that Monk3 has the highest test accuracy. 
    This is due to that the binary splits work effectively here.
    Condition: ( a5 = 1 AND a4 = 1 ) OR ( a5 != 4 AND a2 != 3 )
    If a5 or a4 is 1, the chance greatly increases that we have a positive sample.
    Same for if a5 is not 4 and a2 is not 3.

    It is also logical that Monk1 has the middle-best accuracy.
    Condition: ( a1 = a2 ) OR ( a5 = 1 )
    If a5 is 1, then the sample is positive.
    However,  a1 = a2 is much harder to solve using a binary split.

Training Dataset: Monk1.
     - Test Accuray: 0.8287.
     - Training Accuracy: 1.0

Training Dataset: Monk2.
     - Test Accuray: 0.69213.
     - Training Accuracy: 1.0

Training Dataset: Monk3.
     - Test Accuray: 0.94444.
     - Training Accuracy: 1.0
"""