import DecisionTree
import numpy as np

training_data = [
    # 'Outlook', 'Humidity', 'Wind', 'Play/NotPlay'
    ['Sunny', 'High', 'Weak', 0],
    ['Sunny', 'High', 'Strong', 0],
    ['Overcast', 'High', 'Weak', 1],
    ['Rain', 'High', 'Weak', 1],
    ['Rain', 'Normal', 'Weak', 1],
    ['Rain', 'Normal', 'Strong', 0],
    ['Overcast', 'Normal', 'Strong', 1],
    ['Sunny', 'High', 'Weak', 0],
    ['Sunny', 'Normal', 'Weak', 1],
    ['Rain', 'Normal', 'Weak', 1],
    ['Sunny', 'Normal', 'Strong', 1],
    ['Overcast', 'High', 'Strong', 1],
    ['Overcast', 'Normal', 'Weak', 1]
]
test_data = [
    ['Rain', 'High', 'Strong']  # True Class = 0
]
num_trees = 300
trees = []
predictions = {
    '0': 0,
    '1': 0
}
for k in range(num_trees):
    data = []
    indexes = [np.random.randint(1, 13) for i in range(np.random.randint(5, 13))]
    for index in indexes:
        data.append(training_data[index])
    tree = DecisionTree.make_tree(data)
    trees.append(tree)
for k in range(num_trees):
    y_pred = DecisionTree.classify(test_data[0], trees[k])
    if y_pred == 0:
        predictions['0'] += 1
    elif y_pred == 1:
        predictions['1'] += 1
print('Votes : ', predictions)

if predictions['1'] > predictions['0']:
    print('Final Prediction : 1')
elif predictions['0'] > predictions['1']:
    print('Final Prediction : 0')
else:
    print('Equal Votes Try Again')
