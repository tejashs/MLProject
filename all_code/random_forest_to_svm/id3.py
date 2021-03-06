"""
CODE USED FROM HOMEWORK SOLUTIONS UPLOADED ON CANVAS
HAVE ALTERED IT FOR PROJECT
"""

import math
import random
import numpy as np
import featurization
import svm_sci as SVM

PREDICTION_PATH = "../../data/data-splits/"
OUTPUT_FILE_NAME = "RandomForest_Transformed.data"
f_train = "../../data/data-splits/data.train1.transformed"
f_test = "../../data/data-splits/data.test1.transformed"
f_eval = "../../data/data-splits/data.eval.anon1.transformed"
f_eval_id = "../../data/data-splits/data.eval.id"
NUM_OF_TREES = 1000
NUM_OF_EXAMPLES = 1000
POS_LABEL = 1
NEG_LABEL = 0

depths = {1, 2, 3, 4, 5, 10, 15, 20}


def main():
    # The file contains information about the features
    # Format -> Feature name:Values it can take (seperated by commas)
    with open("feature.info") as f:
        data_info = f.readlines()
    # Create feature nodes
    features = feature_info(data_info)

    # Cross Validation
    # print("Running Cross Validation For depths...")
    # for depth in depths:
    #     cross_validation(depth, features)

    # Transform the data
    with open(f_train) as f:
        data = [line.rstrip() for line in f]
    data_train = featurization.featurize(data)

    with open(f_test) as f:
        tdata = [line.rstrip() for line in f]
    data_test = featurization.featurize(tdata)

    forest = []
    print("Building " + str(NUM_OF_TREES) + " Decision Trees as part of Random Forest")
    for i in range(NUM_OF_TREES):
        random_subset = get_random_examples(data_train, NUM_OF_EXAMPLES)
        tree = build_tree(random_subset, features, -1)
        forest.append(tree)

    new_train_data = test(forest, data_train)
    write_output(new_train_data, PREDICTION_PATH+OUTPUT_FILE_NAME+".train")

    with open(f_eval) as f:
        tdata = [line.rstrip() for line in f]
    eval_data = featurization.featurize(tdata)
    new_eval_data = test(forest, eval_data)
    write_output(new_eval_data, PREDICTION_PATH + OUTPUT_FILE_NAME + ".eval")
    print("Random Forest Feature Transformation Done.")
    #Call SVM
    # print("Feeding Random Forest output into Bagged SVM")
    # gamma = 0.001
    # const = 10
    # print("Gamma - " + str(gamma) + " Constant - " + str(const))
    # svm = SVM.svm()
    # svm.set_output_file_name("random_forest_svm_bagged_gamma_" + str(gamma) + "_const_" + str(const) + ".csv")
    # svm.set_eval_entries(new_eval_data)
    # svm.enter_svm(new_train_data, new_eval_data, gamma, const, True, 10)
    # print ("Random Forest into SVM - Output Generated")


def test(forest, data_test):
    new_data = []
    for d in data_test:
        prediction_tokens = forest_predict(forest, d, False)
        new_data.append(prediction_tokens)
    return new_data


def forest_predict(forest, example, cumulative):
    pos = 0
    total = len(forest)
    prediction_arr = None
    for i in range(total):
        tree = forest[i]
        prediction = walk_down(tree, example[0], example[1])
        if cumulative:
            pos += prediction
        else:
            if prediction_arr:
                prediction_arr += " "
            else:
                temp = example[1] if example[1] == 1 else -1
                prediction_arr = str(temp) +" "
            flag = prediction if prediction == 1 else -1
            prediction_arr += str(i+1) + ":" + str(flag)

    if cumulative:
        if pos > total-pos:
            return POS_LABEL
        else:
            return NEG_LABEL
    else:
        return prediction_arr

def write_output(output_file, file_name):
    f = open(file_name, "w")
    for item in output_file:
        f.write("%s\n" % item)
    print (" --------------------------------------------------")
    print("IDs Output File generated " + str(PREDICTION_PATH + OUTPUT_FILE_NAME))
    print (" --------------------------------------------------")


def build_tree(data_train, features, depth):
    root = ID3(data_train, features, 0, depth)
    return root


def walk_down(node, point, label):
    if node.name == "leaf":
        if node.value == label:
            return 1
    if node.branches:
        for b in node.branches:
            if b.value == point[node.index]:
                return walk_down(b.child, point, label)
    return 0


def ID3(data_samples, attributes, depth, depth_limit):
    if not attributes or depth == depth_limit:
        leaf = Node()
        leaf.set_is_leaf(most_common(data_samples))
        return leaf

    if (all_same(data_samples)):
        label = data_samples[0][1]
        root = Node()
        root.set_is_leaf(label)
        return root

    base_entropy = calculate_base_entropy(data_samples)
    root = best_attribute(data_samples, base_entropy, attributes)
    root = Node(root.name, root.possible_vals, root.index)
    depth += 1

    for val in root.possible_vals:
        b = Branch(val)
        root.add_branch(b)
        subset = subset_val(data_samples, root.index, val)
        if not subset:
            leaf = Node()
            leaf.set_is_leaf(most_common(data_samples))
            b.set_child(leaf)
        else:
            attributes = remove_attribute(attributes, root)
            b.set_child(ID3(subset, attributes, depth, depth_limit))
    return root


def best_attribute(data, base_entropy, attributes):
    max_ig = 0
    max_a = None
    for a in attributes:
        tmp_ig = base_entropy - expected_entropy(data, a)
        tmp_a = a
        if tmp_ig >= max_ig:
            max_ig = tmp_ig
            max_a = a
    return max_a


# Returns the most common label
def most_common(data_samples):
    p = sum(d[1] for d in data_samples)
    if p >= len(data_samples) / 2:
        return 1
    else:
        return 0


def expected_entropy(data, attribute):
    data_total = float(len(data))
    e_entropy = 0.0
    for val in attribute.possible_vals:
        entropy, total = calculate_entropy(data, attribute, val)
        e_entropy += (total / data_total) * entropy
    return e_entropy


def calculate_entropy(data, attribute, value):
    subset = subset_val(data, attribute.index, value)
    if not subset:
        return [0, 0]

    return [calculate_base_entropy(subset), len(subset)]


def calculate_base_entropy(data):
    l = len(data)
    p = sum(d[1] for d in data)

    if not p or l == p:
        return 0

    n = l - p

    probP = p / float(l)
    probN = n / float(l)

    return (-probP * math.log(probP)) - (probN * math.log(probN))


# Returns a subset of the data where the given feature has the given value
def subset_val(data, feature_index, val):
    return [d for d in data if d[0][feature_index] == val]


# Returns true if all the labels are the same in the sample data
def all_same(data_samples):
    label = data_samples[0][1]
    for s in data_samples:
        if s[1] != label:
            return False
    return True


def remove_attribute(attributes, attribute):
    new_attributes = []
    for a in attributes:
        if a.name != attribute.name:
            new_attributes.append(a)
    return new_attributes


def feature_info(data):
    data_inf = []
    for i, d in enumerate(data):
        d = d.split(":")
        r = list(map(int, d[1].rstrip().split(",")))
        a = Node(d[0], r, i)
        data_inf.append(a)

    return data_inf


def rand(max):
    num = int(math.ceil(random.uniform(0, max - 1)))
    return num


def get_random_examples(examples, num_of_exs):
    examples_to_return = []
    total = len(examples)
    for i in range(num_of_exs):
        rand_num = rand(total)
        examples_to_return.append(examples[rand_num])
    return examples_to_return


class Node:
    def __init__(self, name="leaf", vals=None, index=-1):
        self.name = name
        self.possible_vals = vals
        self.index = index
        self.branches = []

    def set_is_leaf(self, value):
        self.leaf = True
        self.value = value

    def add_branch(self, b):
        self.branches.append(b)


class Branch:
    def __init__(self, value):
        self.value = value

    def set_child(self, child):
        self.child = child


if __name__ == '__main__':
    print("Run From Main_forest.py")
