import numpy as np
import random
import math

LEARNING_RATES = [10, 1, 0.1, 0.01, 0.001, 0.0001]
LOSS_PARAMS = [10, 1, 0.1, 0.01, 0.001, 0.0001]
NUM_OF_FEATURES = 16
RANDOM_SAMPLES = 1000
NUM_OF_PREDICTORS = 100
EPOCHS = 10
POS_LABEL = 1
NEG_LABEL = -1
arr = [1] * NUM_OF_FEATURES
x_arr = np.array(arr, dtype=np.float64)
arr2 = [0.01] * NUM_OF_FEATURES
init_weights = np.array(arr2, dtype=np.float64)
RAND_FEATURE_VAL = 0.01
TRAIN_F = "../data/data-splits/data.train"
TEST_F = "../data/data-splits/data.test"


class svm:

    def enter_svm(self, train_entries, test_entries, gamma_t, constant):
        weight_arrays = []
        bias_arrays = []
        print ("Training " + str(NUM_OF_PREDICTORS) + " SVM predictors")
        for i in range(NUM_OF_PREDICTORS):
            weights, bias, train_acc = self.get_trained_svm(gamma_t, constant, train_entries)
            weight_arrays.append(weights)
            bias_arrays.append(bias)

        test_correct = 0
        for line in test_entries:
            prediction, true_label = self.predict_example(line, weight_arrays, bias_arrays)
            if prediction == true_label:
                test_correct += 1

        test_acc = test_correct / float(len(test_entries))
        print("Test Accuracy : " + str(test_acc))
        return test_acc

    def predict_example(self, example, weight_arrays, bias_arrays):
        pos_labels, neg_labels = 0, 0
        for i in range(len(weight_arrays)):
            weights = weight_arrays[i]
            bias = bias_arrays[i]
            features, true_label, xarr = self.get_features_for_example(example)
            # dot product
            dotp = np.dot(weights, xarr)
            sgd = dotp + bias
            if sgd < 0:
                neg_labels += 1
            else:
                pos_labels += 1
        if pos_labels > neg_labels:
            return POS_LABEL, true_label
        else:
            return NEG_LABEL, true_label

    def get_trained_svm(self, rate, constant, all_examples):
        bias = 0.1
        weights = init_weights.copy()
        total_exs = 0
        train_correct = 0
        initial_learning_rate = rate
        gamma_t = rate
        for epoch in range(EPOCHS):
            rand_exs = self.get_random_examples(all_examples, RANDOM_SAMPLES)
            for line in rand_exs:
                total_exs += 1
                features, true_label, xarr = self.get_features_for_example(line)
                # dot product
                dotp = np.dot(weights, xarr)
                sgd = (dotp * true_label) + bias
                weights = weights * (1 - gamma_t)
                bias = bias * (1 - gamma_t)
                if sgd <= 1:
                    rhs = gamma_t * constant * true_label
                    bias = bias + rhs
                    for i in range(len(xarr)):
                        weights[i] = weights[i] + (rhs * xarr[i])
                    continue
                train_correct += 1
            gamma_t = initial_learning_rate / (1 + (initial_learning_rate * epoch / float(constant)))

        train_acc = train_correct / float(total_exs)
        # print("Train Accuracy : " + str(train_acc))
        return weights, bias, train_acc

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
            # if i == 3 or i == 4 or i == 5 or i == 5:
            #     xval = 0
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

    def get_best_learning_rate(self, learning_rates, constants):
        accuracy_list = []
        for constant in constants:
            for gamma_t in learning_rates:
                examples = svm.get_file_entries(TRAIN_F)
                total = len(examples)
                quarter = int(total/4)
                train_exs = examples[: total-quarter-2]
                test_exs = examples[total-quarter-1:]
                acc = self.enter_svm(train_exs, test_exs, gamma_t, constant)
                accuracy_list.append((constant, gamma_t, acc))

        print accuracy_list


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
    svm = svm()
    # svm.get_best_learning_rate(LEARNING_RATES, LOSS_PARAMS)
    # [(10, 10, 0.47735095690486024), (10, 1, 0.8273440726972325), (10, 0.1, 0.7965028225251274),
    #  (10, 0.01, 0.8197714443067603), (10, 0.001, 0.8811785763458626), (10, 0.0001, 0.8694754233787692),
    #  (1, 10, 0.47735095690486024), (1, 1, 0.8579099545642297), (1, 0.1, 0.8110973426958558),
    #  (1, 0.01, 0.872366790582404), (1, 0.001, 0.8711276332094176), (1, 0.0001, 0.8303731240534215),
    #  (0.1, 10, 0.47735095690486024), (0.1, 1, 0.8606636376153105), (0.1, 0.1, 0.8309238606636377),
    #  (0.1, 0.01, 0.8672724769379044), (0.1, 0.001, 0.8654825829547019), (0.1, 0.0001, 0.855569323970811),
    #  (0.01, 10, 0.47735095690486024), (0.01, 1, 0.8777364725320116), (0.01, 0.1, 0.8580476387167837),
    #  (0.01, 0.01, 0.8690623709211069), (0.01, 0.001, 0.8747074211758227), (0.01, 0.0001, 0.8587360594795539),
    #  (0.001, 10, 0.47735095690486024), (0.001, 1, 0.8466198540547983), (0.001, 0.1, 0.8440038551562715),
    #  (0.001, 0.01, 0.8328514387993942), (0.001, 0.001, 0.8433154343935013), (0.001, 0.0001, 0.7857634586259121),
    #  (0.0001, 10, 0.47735095690486024), (0.0001, 1, 0.8561200605810271), (0.0001, 0.1, 0.8467575382073523),
    #  (0.0001, 0.01, 0.8535040616825004), (0.0001, 0.001, 0.8474459589701225), (0.0001, 0.0001, 0.7674514663362247)]

    gamma = 0.001
    const = 10
    train_exs = svm.get_file_entries(TRAIN_F)
    test_exs = svm.get_file_entries(TEST_F)
    svm.enter_svm(train_exs, test_exs, gamma, const)
