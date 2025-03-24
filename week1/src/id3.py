from .utils import entropy, unique_values, split_dataset, most_common_label


class Node:
    def __init__(self, column=None, value=None, true_branch=None, false_branch=None, label=None):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.label = label

    def is_leaf(self):
        return self.label is not None


def best_split(data, target):
    base_entropy = entropy(data, target)
    best_gain = 0
    best_column = None
    for column in data.columns.drop(target):
        values = unique_values(data, column)
        for value in values:
            true_branch, false_branch = split_dataset(data, column, value)
            if len(true_branch) == 0 or len(false_branch) == 0:
                continue
            weight = len(true_branch) / len(data)
            gain = base_entropy - (weight * entropy(true_branch, target) + (1 - weight) * entropy(false_branch, target))
            if gain > best_gain:
                best_gain = gain
                best_column = (column, value)
    return best_column


def build_tree(data, target):
    if len(data[target].unique()) == 1:
        return Node(label=data[target].iloc[0])
    if data.shape[1] == 1:
        return Node(label=most_common_label(data, target))
    best_column = best_split(data, target)
    if not best_column:
        return Node(label=most_common_label(data, target))
    column, value = best_column
    true_branch, false_branch = split_dataset(data, column, value)
    true_node = build_tree(true_branch, target)
    false_node = build_tree(false_branch, target)
    return Node(column=column, value=value, true_branch=true_node, false_branch=false_node)


def predict(node, sample):
    if node.is_leaf():
        return node.label
    if sample[node.column] == node.value:
        return predict(node.true_branch, sample)
    else:
        return predict(node.false_branch, sample)
