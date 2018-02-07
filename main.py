import scipy.io as spio
from tree import TreeNode
import math
import random
import numpy as np


def main():
    clean_data = 'Data/cleandata_students.mat'
    noisy_data = 'Data/noisydata_students.mat'
    total_attributes = 45
    number_of_emotions = 6
    number_of_trees = number_of_emotions
    k_folds = 10
    randomise = False

    # Extract data from the file
    mat = spio.loadmat(clean_data, squeeze_me=True)
    x = list(mat['x'])
    y = list(mat['y'])

    # Create attributes
    attributes = [0] * total_attributes
    for i in range(total_attributes):
        attributes[i] = i + 1
    perc_acc = [0] * k_folds
    # k-fold cross-validation
    for i in range(k_folds):
        test_data_input = []
        test_data_output = []
        traing_data_input = []
        traing_data_output = []
        validation_data_input = []
        validation_data_output = []
        for j in range(len(x)):
            if (j % k_folds == i):
                test_data_input.append(x[j])
                test_data_output.append(y[j])
            elif (j % k_folds == (i+1) % k_folds):
                validation_data_input.append(x[j])
                validation_data_output.append(y[j])
            else:
                traing_data_input.append(x[j])
                traing_data_output.append(y[j])


        tree_priority = [0] * number_of_trees
        if randomise:
            unvalidated_trees = train_trees(number_of_trees, attributes, traing_data_input, traing_data_output)
            tree_priority = get_tree_priority(unvalidated_trees, validation_data_input, validation_data_output)
        trees = train_trees(number_of_trees, attributes, traing_data_input + validation_data_input, traing_data_output + validation_data_output)
        predictions = test_trees(trees, test_data_input, tree_priority, randomise)
        perc_acc[i] = evaluate_results(predictions, test_data_output)
    print(np.mean(perc_acc))
def robust_validation(number_of_trees, trees, x, y):
    trees_FP = [0] * number_of_trees
    trees_FN = [0] * number_of_trees
    for i in range(len(x)):
        for t in range(number_of_trees):
            tree_output = trees[t].parse_tree(x[i])
            if y[i] == t+1 and not tree_output:
                trees_FN[t] += 1
            elif y[i] != t+1 and tree_output:
                trees_FP[t] += 1

def evaluate_results(predictions, actual_outputs):
    correct_cases = 0
    incorrect_cases = 0
    for k in range(len(predictions)):
        if (predictions[k] == actual_outputs[k]):
            correct_cases += 1
        else:
            incorrect_cases += 1
    total = correct_cases + incorrect_cases
    perc_correct = (correct_cases / total) * 100
    perc_incorrect = (incorrect_cases / total) * 100
    return perc_correct

def train_trees(number_of_trees, attributes, traing_data_input, traing_data_output):
    trees = []
    # Parent call to recursive function
    for i in range(1, number_of_trees + 1):
        # the binary target for index i
        y_tree = list(map(lambda value: value == i if 1 else 0, traing_data_output))
        trees.append(decision_tree_learning(traing_data_input, attributes, y_tree))
    return trees

def get_tree_priority(trees, validation_data_input, validation_data_output):
    number_of_trees = len(trees)
    tree_priority = [0] * number_of_trees
    for t in range(number_of_trees):
        tree = trees[t]
        tree_priority[t] = get_perc_accuracy(tree, t + 1, validation_data_input, validation_data_output)
    return tree_priority

def get_perc_accuracy(tree, emotion_val, x, y):
    correct = 0
    for i in range(len(x)):
        output = tree.parse_tree(x[i])
        if (y[i] == emotion_val and output) or (y[i] !=emotion_val and (not output)):
            correct += 1
    return correct / len(x)

def test_trees(trees, test_data, tree_priority, randomise):
    number_of_trees = len(trees)
    final_result = [0] * len(test_data)
    for i in range(len(test_data)):
        test_case_output = [0] * number_of_trees
        for t in range(number_of_trees):
            output = trees[t].parse_tree(test_data[i])
            test_case_output[t] = output
        if randomise:
            final_result[i] = get_emotion_val_rand(test_case_output)
        else:
            final_result[i] = get_emotion_val(test_case_output, tree_priority)
    return final_result

def get_emotion_val_rand(output):
    trues = []
    for i in range(len(output)):
        if output[i]:
            trues.append(i)
    if len(trues) == 0:
        return random.randint(1, len(output))
    return random.choice(trues) + 1

def get_emotion_val(output, tree_priority):
    trues = []
    for i in range(len(output)):
        if output[i]:
            trues.append(tree_priority[i])
        else:
            trues.append(-tree_priority[i])
    return trues.index(max(trues)) + 1

def decision_tree_learning(examples, attributes, binary_targets):
    if same_binary_targets(binary_targets):
        return TreeNode.create_leaf(binary_targets[0])
    elif len(attributes) == 0:
        return TreeNode.create_leaf(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = TreeNode.create_internal(best_attribute)
        for v in [0, 1]:
            v_examples = []
            v_binary_targets = []
            for i in range(0, len(examples)):
                example = examples[i]
                if example[best_attribute - 1] == v:
                    v_examples.append(example)
                    v_binary_targets.append(binary_targets[i])

            if len(v_examples) == 0:
                return TreeNode.create_leaf(majority_value(binary_targets))
            else:
                attributes.remove(best_attribute)
                subtree = decision_tree_learning(v_examples, attributes, v_binary_targets)
                tree.add_kid(subtree)
                attributes.append(best_attribute)
        return tree

# checks if binary_targets vector contains same values
def same_binary_targets(binary_targets):
    if len(binary_targets) <= 0:
        return True
    first_target = binary_targets[0]
    for target in binary_targets:
        if target != first_target:
            return False
    return True


# finds the mode of the vector
def majority_value(binary_targets):
    return max(set(binary_targets), key=binary_targets.count)


def get_entropy(p, n):
    if p == 0 or n == 0:
        return 0

    return -(p/(p+n)) * math.log(p/(p+n), 2) - (n/(p+n)) * math.log(n/(p+n), 2)


# find between 1 and 45
def choose_best_decision_attribute(examples, attributes, binary_targets):
    max_gain = -1
    index_max = -1

    for index, attr in enumerate(attributes):
        p0 = 0
        n0 = 0
        p1 = 0
        n1 = 0
        p = 0
        n = 0

        for i in range(0, len(examples)):
            example = examples[i]
            if example[attr - 1] == 0:
                if binary_targets[i] == 1:
                    p0 += 1
                    p += 1
                else:
                    n0 += 1
                    n += 1
            else:
                if binary_targets[i] == 1:
                    p1 += 1
                    p += 1
                else:
                    n1 += 1
                    n += 1

        remainder = ((p0 + n0) / (p + n)) * get_entropy(p0, n0) + \
                    ((p1 + n1) / (p + n)) * get_entropy(p1, n1)
        gain = get_entropy(p, n) - remainder
        if gain > max_gain:
            max_gain = gain
            index_max = index

    if index_max == -1:
        raise ValueError("Index is -1")

    return attributes[index_max]


def dump_tree(tree_name, tree):
    print("-------------------------------------------------------------------")
    print("Tree Name: " + str(tree_name))
    print(tree.to_string())
    print("-------------------------------------------------------------------")


def test_debug():
    # 1
    # |-2
    # |-3
    # | |-4
    # |   |-5
    # |-6
    tree_test_case = \
        TreeNode.create_internal("1").add_kid(TreeNode.create_leaf("2")) \
            .add_kid(TreeNode.create_internal("3").add_kid(TreeNode.create_internal("4").add_kid(TreeNode.create_leaf("5")))) \
            .add_kid(TreeNode.create_leaf("6"))
    dump_tree("Test", tree_test_case)


if __name__ == "__main__":
    main()
    # test_debug()
