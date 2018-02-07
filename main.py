import scipy.io as spio
from tree import TreeNode
import math


def main():
    clean_data = 'Data/cleandata_students.mat'
    noisy_data = 'Data/noisydata_students.mat'
    total_attributes = 45
    number_of_trees = 6

    # Extract data from the file
    mat = spio.loadmat(clean_data, squeeze_me=True)
    x = list(mat['x'])
    y = list(mat['y'])

    # Create attributes
    attributes = [0] * 45
    for i in range(total_attributes):
        attributes[i] = i + 1

    # trees = []
    # # Parent call to recursive function
    # for i in range(1, number_of_trees + 1):
    #     # the binary target for index i
    #     y_tree = list(map(lambda value: value == i if 1 else 0, y))
    #     trees.append(decision_tree_learning(x, attributes, y_tree))
    #
    # # analyse_data(number_of_trees, x, y)
    # robust_validation(number_of_trees, trees, x, y)
    validation(number_of_trees, attributes, x, y)


def analyse_data(number_of_trees, x, y):
    emotions_count = [0] * 6
    for i in range(len(x)):
        emotions_count[y[i] - 1] += 1
    print(str(emotions_count))


def robust_validation(number_of_trees, trees, x, y):
    trees_FP = [0] * number_of_trees
    trees_FN = [0] * number_of_trees
    for i in range(len(x)):
        for t in range(number_of_trees):
            actual = trees[t].parse_tree(x[i])
            if y[i] == t+1 and not actual:
                # print("Failed FN: " + str(x[i]))
                # print("For emotion: " + str(y[i]))
                trees_FN[t] += 1
            elif y[i] != t+1 and actual:
                # print("Failed FP: " + str(x[i]))
                # print("For emotion: " + str(y[i]))
                trees_FP[t] += 1

    print("FN: " + str(trees_FN))
    print("FP: " + str(trees_FP))
    print()

    # dump_tree("6", trees[5])

def validation(number_of_trees, attributes, x, y):
    for i in range(10):
        test_data_input = []
        test_data_output = []
        traing_data_input = []
        traing_data_output = []
        for j in range(len(x)):
            if (j % 10 == i):
                test_data_input.append(x[j])
                test_data_output.append(y[j])
            else:
                traing_data_input.append(x[j])
                traing_data_output.append(y[j])
        trees = []
        # Parent call to recursive function
        for i in range(1, number_of_trees + 1):
            # the binary target for index i
            y_tree = list(map(lambda value: value == i if 1 else 0, traing_data_output))
            trees.append(decision_tree_learning(traing_data_input, attributes, y_tree))

        robust_validation(number_of_trees, trees, test_data_input, test_data_output)



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
