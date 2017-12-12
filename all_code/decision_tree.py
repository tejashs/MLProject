from __future__ import print_function
import math
import random

POS_LABEL = 1
NEG_LABEL = 0
DEPTH = 10
SAMPLE_NUM = 100
SP = " "

CROSS_VALIDATION_FILES = ["data/training00.data", "data/training01.data", "data/training02.data",
                          "data/training03.data", "data/training04.data"]
F_TRAIN = "data/speeches.train.liblinear"
F_TEST = "data/speeches.test.liblinear"

random.seed(5)


class Decision_Tree:

    def get_label_counts_and_features(self, file_entries, feature_nodes):
        y = dict()
        y[POS_LABEL] = []
        y[NEG_LABEL] = []
        features_dict = dict()
        for line in file_entries:
            tokens = line.split(" ")
            true_label = tokens[0]
            if int(true_label) == POS_LABEL:
                y[POS_LABEL].append(line)
            else:
                y[NEG_LABEL].append(line)

            for token in tokens:
                subtokens = token.split(":")
                if len(subtokens) == 1:
                    continue
                else:
                    feature_name = subtokens[0]
                    feature_val = subtokens[1]
                    if feature_name in features_dict:
                        feature_node = features_dict[feature_name]
                    else:
                        feature_node = dict()
                        features_dict[feature_name] = feature_node

                    if feature_val in feature_node:
                        subset_exs_dict = feature_node[feature_val]
                    else:
                        subset_exs_dict = dict()
                        subset_exs_dict[POS_LABEL] = []
                        subset_exs_dict[NEG_LABEL] = []
                        feature_node[feature_val] = subset_exs_dict

                    current_subset_exs = subset_exs_dict[POS_LABEL] if int(true_label) == POS_LABEL else subset_exs_dict[
                        NEG_LABEL]
                    current_subset_exs.append(line)

        for key in feature_nodes.keys():
            f = feature_nodes[key]
            if f.get_feature in features_dict:
                feature = features_dict[f.get_feature()]
            else:
                continue

            for val in f.get_possible_vals():
                if val in feature:
                    continue
                else:
                    feature[val] = []
        return y, features_dict

    def start(self, num_of_trees, train_file, test_file):
        with open("feature.info") as f:
            feature_info = f.readlines()

        feature_nodes = dict()
        for f in feature_info:
            f = f.rstrip(" ").rstrip("\n").rstrip("")
            if len(f) == 0:
                continue
            tokens = f.split(":")
            feature = tokens[0]
            subtokens = tokens[1].split(",")
            feature_nodes[feature] = FeatureNode(feature, subtokens)

        file_entries = self.get_file_entries(train_file)
        root = self.id3(file_entries, feature_nodes, None, None, depth=0)
        correct, wrong = 0, 0
        for example in file_entries:
            prediction, true_label = self.predict(root,example, None, None)
            if int(prediction) == int(true_label):
                correct +=1
            else:
                wrong +=1

        print("Train Accuracy" + str(correct/float(correct+wrong)))

        test_entries = self.get_file_entries(test_file)
        correct, wrong = 0, 0
        for example in test_entries:
            prediction, true_label = self.predict(root, example, None, None)
            if int(prediction) == int(true_label):
                correct += 1
            else:
                wrong += 1

        print("Test Accuracy" + str(correct / float(correct + wrong)))


    def predict(self, root, example, true_label, features_dict):
        if not true_label and not features_dict:
            true_label, features_dict = self.get_true_label_feature_dict(example)

        if root.is_leaf():
            return root.get_prediction(), true_label
        else:
            feature_name = root.get_feature_name()
            val = features_dict[feature_name]
            child = root.get_branch(val)
            return self.predict(child, example, true_label, features_dict)

    def get_true_label_feature_dict(self, example):
        tokens = example.split(" ")
        features_dict = dict()
        true_label = None
        for token in tokens:
            subtokens = token.split(":")
            if len(subtokens) == 1:
                true_label = subtokens[0]
                continue
            features_dict[subtokens[0]] = subtokens[1]
        return true_label, features_dict


    def calculate_expected_entropy(self, y, feature_node, feature):
        entropy_val_dict = dict()
        full_total_examples = 0
        for val in feature_node.get_possible_vals():
            if val in feature:
                subset_exs_dict = feature[val]
            else: continue
            pos_exs = subset_exs_dict[POS_LABEL]
            neg_exs = subset_exs_dict[NEG_LABEL]
            total = float(len(pos_exs) + len(neg_exs))
            full_total_examples += total
            p = len(pos_exs) / total
            n = len(neg_exs) / total
            if p == 1.0 or n == 1.0:
                entropy = 0.0
            else:
                entropy = - p * self.log2(p) - n * self.log2(n)
            entropy_val_dict[val] = (entropy, total)

        expected_entropy = 0.0
        for val in feature_node.get_possible_vals():
            if val in entropy_val_dict:
                entropy_val = entropy_val_dict[val]
            else: continue
            expected_entropy += (entropy_val[0] * (entropy_val[1] / float(full_total_examples)))

        return expected_entropy

    def get_best_feature_name(self, y, feature_nodes, features_dict):
        entropy_dict = dict()
        for key in feature_nodes.keys():
            f = feature_nodes[key]
            f_name = f.get_feature()
            expected_entropy = self.calculate_expected_entropy(y, f, features_dict[f_name])
            entropy_dict[f_name] = expected_entropy
        # Return key with lowest entropy, hence max gain
        v = list(entropy_dict.values())
        k = list(entropy_dict.keys())
        if len(v) == 0:
            print("WTF")
        best_feature = k[v.index(min(v))]
        return best_feature


    def id3(self, examples, feature_nodes, features_dict, y, depth):
        if not features_dict:
            y, features_dict = self.get_label_counts_and_features(examples, feature_nodes)
        else:
            y, _ = self.get_label_counts_and_features(examples, feature_nodes)

        if len(feature_nodes) == 0 or depth == DEPTH:
            node = Node(None, depth)
            node.set_leaf(True)
            label = POS_LABEL if len(y[POS_LABEL]) > len(y[NEG_LABEL]) else NEG_LABEL
            node.set_prediction_label(label)
            return node

        best_feature_name = self.get_best_feature_name(y, feature_nodes, features_dict)
        features_dict_copy = features_dict.copy()
        feature_nodes_copy = feature_nodes.copy()
        features_dict_copy.pop(best_feature_name)
        feature_nodes_copy.pop(best_feature_name)
        feature = features_dict[best_feature_name]
        feature_node = feature_nodes[best_feature_name]
        node = Node(feature_node, depth)
        # Check if feature has same value for all exs
        multiple_vals, unique_val = self.check_feature_multiple_values(feature)
        if not multiple_vals:
            subset_exs = feature[unique_val]
            node.set_leaf(True)
            prediction = POS_LABEL if len(subset_exs[POS_LABEL]) > len (subset_exs[NEG_LABEL]) else NEG_LABEL
            node.set_prediction_label(prediction)
            return node

        for val in feature.keys():
            pos_exs = feature[val][POS_LABEL]
            neg_exs = feature[val][NEG_LABEL]
            if len(pos_exs) == 0:
                child = Node(None, depth+1)
                child.set_leaf(True)
                child.set_prediction_label(NEG_LABEL)
            elif len(neg_exs) == 0:
                child = Node(None, depth+1)
                child.set_leaf(True)
                child.set_prediction_label(POS_LABEL)
            else:
                subset_exs = []
                subset_exs.extend(pos_exs)
                subset_exs.extend(neg_exs)
                child = self.id3(subset_exs, feature_nodes_copy, features_dict_copy, y, depth+1)

            node.add_branch(val, child)

        return node


    def check_feature_multiple_values(self, feature):
        non_zero_vals = 0
        feature_val = None
        for val in feature.keys():
            exs = feature[val]
            if len(exs) > 0:
                non_zero_vals += 1
                feature_val = val
        multiple_vals = False if non_zero_vals < 2 else True
        return multiple_vals, feature_val

    def log2(self, num):
        return math.log(num) / (1.0 * math.log(2))

    def get_file_entries(self, file_name):
        f = open(file_name, "r")
        lines = []
        for line in f:
            line = line.rstrip().rstrip("\n")
            lines.append(line)
        return lines

    def get_random_entries(self, entries, num_of_entries):
        entries_to_return = []
        total_entries = len(entries)
        for i in range(num_of_entries):
            rand_num = self.rand(total_entries)
            entries_to_return.append(entries[rand_num])
        return entries_to_return

    def rand(self, max):
        num = int(math.ceil(random.uniform(0, max - 1)))
        return num



class Node:
    def __init__(self, feature, depth):
        self.feature = feature
        self.depth = depth
        self.branches = dict()
        self.leaf = False
        self.prediction = None

    def add_branch(self, value, branch):
        if branch and value:
            self.branches[value] = branch

    def get_branch(self, value):
        if len(self.branches) == 0 or value not in self.branches:
            return None
        return self.branches[value]

    def get_feature_name(self):
        return self.feature.get_feature()

    def is_leaf(self):
        return self.leaf

    def set_leaf(self, is_leaf):
        self.leaf = is_leaf

    def set_prediction_label(self, val):
        self.prediction = val

    def get_prediction(self):
        return self.prediction

    def get_num_branches(self):
        return len(self.branches)

class FeatureNode:
    def __init__(self, feature, vals=[]):
        self.feature = feature
        self.possible_vals = vals

    def get_possible_vals(self):
        return self.possible_vals

    def get_feature(self):
        return self.feature
