import scipy.io as spio
import os
import argparse
from tree import TreeNode
import math
import random
import numpy as np
import multiprocessing
from functools import partial

# constants
CLEAN_DATA = 'Data/cleandata_students.mat'
NOISY_DATA = 'Data/noisydata_students.mat'
TOTAL_ATTRIBUTES = 45
NUMBER_OF_EMOTIONS = 6
NUMBER_OF_TREES = NUMBER_OF_EMOTIONS
K_FOLDS = 10
RANDOMISE = False


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--load", help="Boolean: Load the trees.", action='store_true')
    group.add_argument("--train", help="Boolean: Load the trees.", action='store_true')

    parser.add_argument("--save", help="Boolean: Train and save the trees.", action='store_true')
    parser.add_argument("--data", help="String: clean or noisy", type=str)
    parser.add_argument("--dump", help="Boolean: Train and save the trees.", action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Extract data from the file
    if args.data is not None and args.data == "noisy":
        mat = spio.loadmat(NOISY_DATA, squeeze_me=True)
    else:
        mat = spio.loadmat(CLEAN_DATA, squeeze_me=True)

    x = list(mat['x'])
    y = list(mat['y'])
    trees = []

    if args.load:
        trees = load_trees(NUMBER_OF_TREES)

    if args.train:
        pool = multiprocessing.Pool(K_FOLDS)
        attributes = list(range(1, TOTAL_ATTRIBUTES + 1))
        train = partial(train_validate, x=x, y=y, attributes=attributes[:], number_of_trees=NUMBER_OF_TREES, k_folds=K_FOLDS, randomise=RANDOMISE)

        # save or interpret result
        res = pool.map(train, range(K_FOLDS))
        mats = list(map(lambda tup: tup[0], res))
        percentages = list(map(lambda tup: tup[1], res))
        all_trees = list(map(lambda tup: tup[2], res))

        trees = choose_best(percentages, all_trees)

        if args.save:
            save_trees(trees)

        confusion_matrix = get_average_confusion_mat(NUMBER_OF_TREES, mats)
        print("Trained Successfully")
        print("Maximum precision: " + str(max(percentages)))
        print("Confusion matrix: " + str(confusion_matrix))

    if args.dump:
        for i in range(len(trees)):
            dump_tree(i, trees[i])


def choose_best(percentages, trees):
    if not os.path.exists("trees"):
        os.makedirs("trees")

    # choose best trees
    res = zip(percentages, trees)
    best_trees = sorted(res, key=lambda el: el[0], reverse=True)[0][1]
    return best_trees


def save_trees(trees):
    for i in range(len(trees)):
        trees[i].save_tree("trees/tree_" + str(i))


def load_trees(nr_of_trees):
    trees = []
    for i in range(nr_of_trees):
        trees.append(TreeNode.load_tree("trees/tree_" + str(i)))
    return trees


def get_average_confusion_mat(number_of_trees, mats):
    # compute average confusion matrix
    confusion_matrix = [[0 for _ in range(number_of_trees)] for _ in range(number_of_trees)]
    for mat in mats:
        for i in range(number_of_trees):
            for j in range(number_of_trees):
                confusion_matrix[i][j] += mat[i][j]

    for i in range(number_of_trees):
        for j in range(number_of_trees):
            confusion_matrix[i][j] /= len(mats)
    return confusion_matrix


def get_confusion_mat(predictions, test_data_output, number_of_trees):
    confusion_mat = [[0 for _ in range(number_of_trees)] for _ in range(number_of_trees)]
    # get confusion matrix
    for i in range(len(predictions)):
        confusion_mat[test_data_output[i] - 1][predictions[i] - 1] += 1
    return confusion_mat


def train_validate(i, x, y, attributes, number_of_trees, k_folds, randomise):
    test_data_input = []
    test_data_output = []
    training_data_input = []
    training_data_output = []
    validation_data_input = []
    validation_data_output = []
    for j in range(len(x)):
        if j % k_folds == i:
            # One fold data used for tests
            test_data_input.append(x[j])
            test_data_output.append(y[j])
        elif j % k_folds == (i+1) % k_folds:
            # One fold data used for validation
            validation_data_input.append(x[j])
            validation_data_output.append(y[j])
        else:
            # Remaining eight fold data used for training
            training_data_input.append(x[j])
            training_data_output.append(y[j])
    tree_priority = [0] * number_of_trees
    if not randomise:
        unvalidated_trees = train_trees(number_of_trees, attributes[:], training_data_input, training_data_output)
        tree_priority = get_tree_priority(unvalidated_trees, validation_data_input, validation_data_output)
    trees = train_trees(number_of_trees, attributes[:], training_data_input + validation_data_input, training_data_output + validation_data_output)
    predictions = test_trees(trees, test_data_input, tree_priority, randomise)

    confusion_mat = get_confusion_mat(predictions, test_data_output, number_of_trees)

    return confusion_mat, evaluate_results(predictions, test_data_output), trees


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
        if predictions[k] == actual_outputs[k]:
            correct_cases += 1
        else:
            incorrect_cases += 1
    total = correct_cases + incorrect_cases
    perc_correct = (correct_cases / total) * 100
    perc_incorrect = (incorrect_cases / total) * 100
    return perc_correct


def train_trees(number_of_trees, attributes, training_data_input, training_data_output):
    trees = []
    # Parent call to recursive function
    for i in range(1, number_of_trees + 1):
        # the binary target for index i
        y_tree = list(map(lambda value: value == i if 1 else 0, training_data_output))
        trees.append(decision_tree_learning(training_data_input, attributes, y_tree))
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
        if (y[i] == emotion_val and output) or (y[i] != emotion_val and (not output)):
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
    maxs = []

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
            maxs = [index]
        elif gain == max_gain:
            maxs.append(index)

    if len(maxs) == 0:
        raise ValueError("Index is -1")

    return attributes[random.choice(maxs)]


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
