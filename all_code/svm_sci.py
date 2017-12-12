import numpy as np
from scipy.sparse import csr_matrix as csr
import random
import math

NUM_OF_FEATURES = 16
EPOCHS = 5
arr = [1] * NUM_OF_FEATURES
x_arr = np.array(arr, dtype=np.float64)
arr2 = [0.01] * NUM_OF_FEATURES
init_weights = np.array(arr2, dtype=np.float64)
RAND_FEATURE_VAL = 0.01
TRAIN_F = "../data/data-splits/data.train"
TEST_F = "../data/data-splits/data.test"


class svm:
    def enter_svm(self, f_train, f_test):
        train_entries = self.get_file_entries(f_train)
        gamma_t = 0.01
        constant = 1
        bias = 0.1
        weights = init_weights.copy()
        train_correct = 0
        total_exs = 0
        initial_learning_rate = gamma_t
        for epoch in range(EPOCHS):
            rand_exs = self.get_random_examples(train_entries, 100)
            for line in rand_exs:
                total_exs += 1
                features, true_label, xarr = self.get_features_for_example(line)
                # dot product
                dotp = np.dot(weights, xarr)
                sgd = dotp * true_label
                weights = weights * (1 - gamma_t)
                if sgd <= 1:
                    rhs = gamma_t * constant * true_label
                    for i in range(len(xarr)):
                        weights[i] = weights[i] + (rhs * xarr[i])
                    continue
                train_correct +=1
            gamma_t = initial_learning_rate / (1 + (initial_learning_rate * epoch / float(constant)))

        train_acc = train_correct / float(total_exs)
        print("Train Accuracy : " + str(train_acc))

        test_entries = self.get_file_entries(f_test)
        test_correct = 0
        for line in test_entries:
            features, true_label, xarr = self.get_features_for_example(line)
            # dot product
            dotp = np.dot(weights, xarr)
            sgd = dotp * true_label
            if sgd > 1:
                test_correct += 1

        test_acc = test_correct / float(len(test_entries))
        print("Test Accuracy : " + str(test_acc))

    def predict(self, weights, xarr, feature_indices, true_label, bias):
        weights_for_features = weights[feature_indices]
        val = np.sum(weights_for_features)
        success = False if (val * true_label + bias) <= 1 else True
        return success, weights_for_features

    def update_weights(self, success, gamma_t, constant, weights, weights_for_features, feature_indices, bias,
                       true_label):
        if success:
            weights = weights * (1 - gamma_t)
            bias = bias * (1 - gamma_t)
        else:
            weights_for_features = weights_for_features * (1 - gamma_t)
            weights_for_features += (gamma_t * constant * true_label)
            weights[feature_indices] = weights_for_features

        return bias, weights

    def get_features_for_example(self, example):
        tokens = example.split(" ")
        features = []
        tempX = []
        true_label = None
        for i in range(len(tokens)):
            if i == 0:
                true_label = int(tokens[i])
                if true_label == 0:
                    true_label = -1
                continue
            subtokens = tokens[i].split(":")
            features.append(int(subtokens[0]))
            xval = float(subtokens[1])
            # if xval == 0:
            #     pass
            # elif xval < 0:
            #     xval = math.fabs(xval)
            #     xval = self.log2(xval)
            #     if not xval < 0:
            #         xval = -1 * xval
            # else:
            #     xval = self.log2(xval)
            tempX.append(xval)
        xarr = np.array(tempX, dtype=np.float64)
        return features, true_label, xarr

    def get_file_entries(self, file_name):
        file = open(file_name, "r")
        entries = []
        for line in file:
            line = line.rstrip().rstrip("\n")
            entries.append(line)
        return entries

    def init_weight_csr_vector(self):
        arr = [RAND_FEATURE_VAL] * NUM_OF_FEATURES
        w_arr = csr(arr, dtype=np.float64)
        return w_arr

    def get_best_learning_rate(self, learning_rates, constants):
        CROSS_VALIDATION_EPOCHS = 10
        accuracy_list = []
        for constant in constants:
            for gamma_t in learning_rates:
                bias = 0.1
                weights = init_weights.copy()
                initial_learning_rate = gamma_t
                total_accuracy = 0.0
                print(
                "-----------------------------------------------------------------------------------------------------------------------")
                for i in range(5):
                    test_file = CROSS_VALIDATION_FILES[i]
                    train_files = [file for file in CROSS_VALIDATION_FILES if file != test_file]
                    for file in train_files:
                        rate = initial_learning_rate
                        entries = self.get_file_entries(file)
                        for epoch in range(CROSS_VALIDATION_EPOCHS):
                            random.shuffle(entries)
                            # CALL
                            for line in entries:
                                feature_indices, true_label, xarr = self.get_features_for_example(line)
                                # dot product
                                success, weights_for_features = self.predict(weights, feature_indices, true_label, bias)
                                bias, weights = self.update_weights(success, gamma_t, constant, weights,
                                                                    weights_for_features,
                                                                    feature_indices, bias, true_label)
                            gamma_t = initial_learning_rate / float(1 + initial_learning_rate * (epoch + 1) / constant)
                    # Cross-validation prediction
                    # PREDICT
                    test_entries = self.get_file_entries(test_file)
                    test_correct = 0
                    for line in test_entries:
                        feature_indices, true_label, xarr = self.get_features_for_example(line)
                        # dot product
                        success, _ = self.predict(weights, feature_indices, true_label, bias)
                        if success:
                            test_correct += 1
                    test_acc = test_correct / float(len(test_entries))
                    print("Running Cross Validation : Test File - " + str(test_file) + " Training Files - " + str(
                        train_files) + "---- Accuracy : " + str(test_acc))
                    total_accuracy += test_acc
                average_accuracy = total_accuracy / float(5)
                accuracy_list.append((constant, initial_learning_rate, average_accuracy))
                print(" CONSTANT : " + str(constant) + " RATE : " + str(initial_learning_rate) + " ACCURACY : " + str(
                    average_accuracy))

        print(
        "-----------------------------------------------------------------------------------------------------------------------")
        return max(accuracy_list, key=lambda item: item[2])

    def log2(self, num):
        return math.log(num, 2)

    def rand(self, max):
        num = int(math.ceil(random.uniform(0, max - 1)))
        return num

    def get_random_examples(self, examples, num_of_exs):
        examples_to_return = []
        total = len(examples)
        for i in range(num_of_exs):
            rand_num = self.rand(total)
            examples_to_return.append(examples[rand_num])
        return examples_to_return



if __name__ == '__main__':
    svm().enter_svm(TRAIN_F, TEST_F)