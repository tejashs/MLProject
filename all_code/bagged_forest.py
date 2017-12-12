from __future__ import print_function
import math
import random

POS_LABEL = 1
NEG_LABEL = -1
DEPTH = 3
SAMPLE_NUM = 100
SP = " "

CROSS_VALIDATION_FILES = ["data/training00.data", "data/training01.data", "data/training02.data",
                          "data/training03.data", "data/training04.data"]
F_TRAIN = "data/speeches.train.liblinear"
F_TEST = "data/speeches.test.liblinear"

random.seed(5)


class Decision_Tree:
    def start(self, num_of_trees):
        file_entries = self.get_file_entries(F_TRAIN)
        forest = self.build_trees(num_of_trees, file_entries)
        self.do_prediction(forest)

    def build_trees(self, num_of_trees, file_entries):
        forest = []
        for i in range(num_of_trees):
            random_entries = self.get_random_entries(file_entries, SAMPLE_NUM)
            y, depth = None, 0
            root = self.id3(random_entries, None, y, depth)
            forest.append(root)
        return forest

    def do_prediction(self, forest):
        file_entries = self.get_file_entries(F_TEST)
        correct, wrong = 0, 0
        for line in file_entries:
            true_label, features_dict = self.get_features_dictionary_and_true_label(line)
            pos_predictions, neg_predictions = 0, 0
            for i in range(len(forest)):
                tree = forest[i]
                leaf = self.tree_walk(tree, features_dict)
                prediction = leaf.get_value()
                if prediction == POS_LABEL:
                    pos_predictions += 1
                else:
                    neg_predictions += 1

            final_prediction = POS_LABEL if pos_predictions > neg_predictions else NEG_LABEL
            if final_prediction == true_label:
                correct += 1
            else:
                wrong += 1

        accuracy = correct*100 / float(correct + wrong)
        print("Correct: " + str(correct) + " Wrong: " + str(wrong) + " Accuracy : " + str(accuracy))

    def tree_walk(self, node, features_dict):
        feature = node.get_feature()
        if feature == "leaf":
            return node
        if len(features_dict) == 0:
            current_val = POS_LABEL if int(self.rand(2)) == 0 else NEG_LABEL
        else:
            current_val = POS_LABEL if feature in features_dict else NEG_LABEL
        branch = node.get_branch(current_val)
        return self.tree_walk(branch, features_dict)

    def id3(self, examples, previous_feature, y, depth):
        y, features_dict = self.get_label_counts_and_features(examples)
        y_pos, y_neg, total = 0, 0, 0
        if y:
            y_pos, y_neg = len(y[POS_LABEL]), len(y[NEG_LABEL])
            total = y_pos + y_neg

        if depth == DEPTH + 1 or (total != 0 and (total == y_pos or total == y_neg)):
            label = POS_LABEL if y_pos > y_neg else NEG_LABEL
            node = Node("leaf", depth)
            node.set_value(label)
            return node
        else:
            features_dict.pop(previous_feature, None)
            if len(features_dict) == 0:
                label = POS_LABEL
                node = Node("leaf", depth)
                node.set_value(label)
                return node
            entropies_dict = self.calculate_expected_entropies(y, features_dict)
            best_feature = self.get_best_feature(entropies_dict, features_dict, y)
            feature_node = Node(best_feature[0], depth)
            subset_examples = best_feature[1]
            pos_subset_exs = subset_examples[POS_LABEL]
            neg_subset_exs = subset_examples[NEG_LABEL]
            if len(pos_subset_exs) == 0:
                node = Node("leaf", depth)
                node.set_value(POS_LABEL)
                feature_node.add_branch(POS_LABEL, node)
                feature_node.add_branch(NEG_LABEL, self.id3(neg_subset_exs, feature_node, y, depth + 1))
            elif len(neg_subset_exs) == 0:
                node = Node("leaf", depth)
                node.set_value(NEG_LABEL)
                feature_node.add_branch(NEG_LABEL, node)
                feature_node.add_branch(POS_LABEL, self.id3(neg_subset_exs, feature_node, y, depth + 1))
            else:
                feature_node.add_branch(POS_LABEL, self.id3(pos_subset_exs, feature_node, y, depth + 1))
                feature_node.add_branch(NEG_LABEL, self.id3(neg_subset_exs, feature_node, y, depth + 1))

            return feature_node

    def get_label_counts_and_features(self, file_entries):
        y = dict()
        y[POS_LABEL] = []
        y[NEG_LABEL] = []
        # This dict contains another dict with 2 entries. Key with POS_LABEL = file entries with true label = POS.
        # Key with NEG_LABEL = file entries with true label = NEG
        features = dict()
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
                    self.increment_count_in_dict(subtokens[0], true_label, features, line)

        return y, features

    # Depending on the value of true label, increment the value in the list appropriately.
    # If true label is POS, then increment 1st value in list, otherwise 2nd.
    def increment_count_in_dict(self, feature_label, true_label, feature_dict, example):
        if feature_label not in feature_dict:
            feature_val_dict = dict()
            feature_val_dict[POS_LABEL] = []
            feature_val_dict[NEG_LABEL] = []
        else:
            feature_val_dict = feature_dict[feature_label]

        feature_dict[feature_label] = self.increment_count(true_label, feature_val_dict, example)
        return

    # depending on the value increment value in the list appropriately.
    # 1st value in list is positive label count and 2nd is negative
    @staticmethod
    def increment_count(true_label, feature_val_dict, example):
        arr = feature_val_dict[int(true_label)]
        arr.append(example)
        feature_val_dict[int(true_label)] = arr
        return feature_val_dict

    def calculate_expected_entropies(self, y, features_dict):
        entropies = dict()
        total_examples = len(y[POS_LABEL]) + len(y[NEG_LABEL])
        for key in features_dict.keys():
            feature_val_dict = features_dict[key]
            total_fpos = len(feature_val_dict[POS_LABEL]) + len(feature_val_dict[NEG_LABEL])
            total_fneg = (len(y[POS_LABEL]) - len(feature_val_dict[POS_LABEL])) + \
                         (len(y[NEG_LABEL]) - len(feature_val_dict[NEG_LABEL]))

            if total_fpos == len(feature_val_dict[POS_LABEL]) or total_fpos == len(feature_val_dict[NEG_LABEL]):
                pos_f_entropy = 0
            else:
                p1 = len(feature_val_dict[POS_LABEL]) / float(total_fpos)
                n1 = len(feature_val_dict[NEG_LABEL]) / float(total_fpos)
                pos_f_entropy = (-p1 * self.log2(p1)) - (n1 * self.log2(n1))

            if total_fneg == (len(y[POS_LABEL]) - len(feature_val_dict[POS_LABEL])) or \
                            total_fneg == (len(y[NEG_LABEL]) - len(feature_val_dict[NEG_LABEL])):
                neg_f_entropy = 0
            else:
                p2 = (len(y[POS_LABEL]) - len(feature_val_dict[POS_LABEL])) / float(total_fneg)
                n2 = (len(y[NEG_LABEL]) - len(feature_val_dict[NEG_LABEL])) / float(total_fneg)
                neg_f_entropy = (-p2 * self.log2(p2)) - (n2 * self.log2(n2))

            expected_entropy = ((pos_f_entropy * total_fpos) / float(total_examples)) + \
                               ((neg_f_entropy * total_fneg) / float(total_examples))

            entropies[key] = expected_entropy
        return entropies

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

    def get_best_feature(self, entropies_dict, features_dict, y):
        best = sorted(entropies_dict.items(), key=lambda (k, v): (v, k))
        try:
            best = best[0]
        except IndexError:
            print("")
        feature_val_dict = features_dict[best[0]]
        total_feature_pos_exs = []
        total_feature_pos_exs.extend(feature_val_dict[POS_LABEL])
        total_feature_pos_exs.extend(feature_val_dict[NEG_LABEL])
        total_feature_neg_exs = []
        total_feature_neg_exs.extend([x for x in y[POS_LABEL] if x not in total_feature_pos_exs])
        total_feature_neg_exs.extend([x for x in y[NEG_LABEL] if x not in total_feature_pos_exs])
        total_feature_examples = dict()
        total_feature_examples[POS_LABEL] = total_feature_pos_exs
        total_feature_examples[NEG_LABEL] = total_feature_neg_exs
        return [best[0], total_feature_examples]

    def rand(self, max):
        num = int(math.ceil(random.uniform(0, max - 1)))
        return num

    def get_features_dictionary_and_true_label(self, line):
        tokens = line.split(" ")
        true_label = None
        feature_dict = dict()
        for token in tokens:
            subtokens = token.split(":")
            if len(subtokens) == 1:
                true_label = int(subtokens[0])
                continue
            feature_dict[subtokens[0]] = 1
        return true_label, feature_dict


class Node:
    def __init__(self, feature, depth):
        self.feature = feature
        self.depth = depth
        self.branches = dict()

    def add_branch(self, value, branch):
        if branch and value:
            self.branches[value] = branch

    def get_branch(self, value):
        if len(self.branches) == 0:
            return None
        return self.branches[value]

    def set_value(self, value):
        self.value = value

    def get_value(self):
        # For leaf nodes
        return self.value

    def get_feature(self):
        return self.feature
