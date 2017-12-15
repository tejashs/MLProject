import numpy as np
import random
import math

NUM_OF_FEATURES = 1000
RANDOM_SAMPLES = 1000
EPOCHS = 10
POS_LABEL = 1
NEG_LABEL = -1
arr = [1] * NUM_OF_FEATURES
x_arr = np.array(arr, dtype=np.float64)
arr2 = [0.01] * NUM_OF_FEATURES
init_weights = np.array(arr2, dtype=np.float64)
RAND_FEATURE_VAL = 0.01
EVAL_ID = "../../data/data-splits/data.eval.id"
EVAL = "../../data/data-splits/data.eval.anon"
PREDICTION_PATH = "../../data/data-splits/"


class svm:

    def enter_svm(self, train_entries, test_entries, gamma_t, constant, gen_output, num_predictors):
        weight_arrays = []
        bias_arrays = []
        if num_predictors > 1:
            print ("Training " + str(num_predictors) + " SVM predictors")
        for i in range(num_predictors):
            weights, bias, train_acc = self.get_trained_svm(gamma_t, constant, train_entries)
            weight_arrays.append(weights)
            bias_arrays.append(bias)

        test_correct = 0
        for line in test_entries:
            prediction, true_label = self.predict_example(line, weight_arrays, bias_arrays)
            if prediction == true_label:
                test_correct += 1

        test_acc = test_correct / float(len(test_entries))
        print("Random Forest into Bagged SVM - Train Accuracy : " + str(test_acc))

        if gen_output:
            ids = self.get_file_entries(EVAL_ID)
            examples = self.eval_entries
            file_index = 0
            output_file = []
            for ex in examples:
                prediction, _ = self.predict_example(ex, weight_arrays, bias_arrays)
                if prediction < 0:
                    flag = 0
                else:
                    flag = 1
                output_file.append(str(ids[file_index]) + "," + str(flag))
                file_index += 1

            f = open(PREDICTION_PATH + self.output_file_name, "w")
            f.write("Id,Prediction\n")
            for item in output_file:
                f.write("%s\n" % item)
            print (" --------------------------------------------------")
            print("IDs Output File generated " + str(PREDICTION_PATH + self.output_file_name))
            print (" --------------------------------------------------")

        return None

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
            xval = self.transform_log(xval)
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

    def get_best_learning_rate(self, learning_rates, constants, train_file):
        accuracy_list = []
        for constant in constants:
            for gamma_t in learning_rates:
                examples = self.get_file_entries(train_file)
                total = len(examples)
                quarter = int(total/4)
                train_exs = examples[: total-quarter-2]
                test_exs = examples[total-quarter-1:]
                acc = self.enter_svm(train_exs, test_exs, gamma_t, constant, False, 1)
                accuracy_list.append((constant, gamma_t, acc))
        print accuracy_list
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

    def transform_log(self, feature_val):
        if feature_val == 0:
            pass
        elif feature_val < 0 :
            val = math.fabs(feature_val)
            val = self.log2(val)
            feature_val = -val
        else:
            feature_val = self.log2(feature_val)
        return feature_val

    def set_output_file_name(self, output_file_name):
        self.output_file_name = output_file_name
        return

    def set_eval_entries(self, eval_entries):
        self.eval_entries = eval_entries


if __name__ == '__main__':
    print("Run from Main_forest.py")
