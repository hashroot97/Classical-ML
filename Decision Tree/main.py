import operator
import numpy as np


class LeafNode:
    def __init__(self, rows):
        counts = class_count(rows)
        self.pred = max(counts.items(), key=operator.itemgetter(1))[0]


class DecisionNode:
    def __init__(self, ques, l_branch, r_branch):
        self.true_branch = l_branch
        self.false_branch = r_branch
        self.question = ques


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


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def get_answer_abs(row, feature, value):
    if row[feature] == value:
        return True
    else:
        return False


def get_answer_num(row, feature, value):
    if row[feature] >= value:
        return True
    else:
        return False


def get_split_rows(data, feature, value):
    left_rows = []
    right_rows = []
    for r in data:
        if is_numeric(value):
            answer = get_answer_num(r, feature, value)
        else:
            answer = get_answer_abs(r, feature, value)
        if answer:
            left_rows.append(r)
        else:
            right_rows.append(r)
    return left_rows, right_rows


def class_count(rows):
    counts = {}
    for i in range(len(rows)):
        if rows[i][-1] not in counts.keys():
            counts[rows[i][-1]] = 1
        else:
            counts[rows[i][-1]] += 1
    return counts


def impurity(rows):
    counts = class_count(rows)
    impurity = 1
    for clas in counts:
        prob = counts[clas]/len(rows)
        impurity -= prob**2
    return impurity


def info_gain(left, right, data):
    p = float(len(left)) / (len(left) + len(right))
    return impurity(data) - p * impurity(left) - (1 - p) * impurity(right)


features = ['Outlook', 'Humidity', 'Wind']


def find_unique_feat(col):
    unq = []
    for val in col:
        if val not in unq:
            unq.append(val)
        else:
            continue
    return unq


def find_best_spilt(data):
    #     print('Starting Data', data, sep='\n')
    gains = []
    ques = []
    true_rows = []
    false_rows = []

    for feat in range(len(data[0])-1):
        unq = find_unique_feat([data[k][feat] for k in range(len(data))])
        for val in unq:
            #             print('Is {} = {}'.format(features[feat], val))
            l_rows, r_rows = get_split_rows(data, feat, val)
            true_rows.append(l_rows)
            false_rows.append(r_rows)

            gain = info_gain(l_rows, r_rows, data)
#             print(gain)

            ques.append([feat, val])
            gains.append(gain)

    best_split = np.argmax(np.asarray(gains))
    return ques[best_split], gains[best_split], true_rows[best_split], \
        false_rows[best_split]


def make_tree(data):

    ques, gain, true_rows, false_rows = find_best_spilt(data)

    if gain == 0:
        return LeafNode(data)
    true_branch = make_tree(true_rows)
    false_branch = make_tree(false_rows)

    return DecisionNode(ques, true_branch, false_branch)


def print_tree(node, spacing=""):

    if isinstance(node, LeafNode):
        print(spacing + "Predict", node.pred)
        return

    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    if isinstance(node, LeafNode):
        return node.pred

    if get_answer_abs(row, node.question[0], node.question[1]):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


tree = make_tree(training_data)
print_tree(tree)
y_pred = classify(test_data[0], tree)

print('Predicted Y : {}'.format(y_pred))
