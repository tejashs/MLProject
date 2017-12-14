import numpy as n
import random
import math

PREDICTION_PATH = "../../data/data-splits/"
TRAIN_F = "../../data/data-splits/data.train"
TEST_F = "../../data/data-splits/data.test"
POS_LABEL = 1
NEG_LABEL = -1
ZERO_LABEL = 0
SIZE_OF_FEATURES = 16
NUM_OF_EPOCHS = 20
DEFAULT_TIME_STEP = 0
SHOULD_PRINT = True


def log(string):
    if SHOULD_PRINT:
        print(string)


def rand():
    return random.uniform(-0.01, 0.01)


class Perceptron:
    def __init__(self):
        self.bias = 0.0
        self.id_file = "../../data/data-splits/data.eval.id"
        self.eval_file = "../../data/data-splits/data.eval.anon"
        f = open(self.id_file, "r")
        self.ids = [x.strip('\n') for x in f]
        self.weight_update_count = 0

    def start_perceptron2(self, learning_rate, is_dynamic_learning, margin, is_aggressive):
        init_array = [rand()] * SIZE_OF_FEATURES
        examples = self.get_file_entries(TRAIN_F)
        weight_vector = n.array(init_array)
        bias = rand()
        initial_learning_rate = learning_rate
        time_step = DEFAULT_TIME_STEP
        for j in range(NUM_OF_EPOCHS):
            rand_entries = self.get_random_examples(examples, 1000)
            learning_rate, accuracy, weight_vector, bias, ts = self.run_base_case_perceptron(rand_entries,
                                                                                             weight_vector, bias,
                                                                                             learning_rate, margin,
                                                                                             False,
                                                                                             is_dynamic_learning,
                                                                                             time_step,
                                                                                             initial_learning_rate,
                                                                                             is_aggressive, None)
            if is_dynamic_learning:
                time_step = ts

        print("Predicting Test Examples...")
        test_examples = self.get_file_entries(TEST_F)
        correct, total = self.predict_set_in_file(test_examples, weight_vector, bias, False)
        log(
            "-----------------------------------------------------------------------------------------------------------------------")
        log(
            "***********************************************************************************************************************")
        log("Test Set Accuracy : " + str(correct * 100 / float(total)))
        log(
            "***********************************************************************************************************************")
        print("Predicting Eval Examples... And Generating Output")
        eval_examples = self.get_file_entries(self.eval_file)
        correct, total = self.predict_set_in_file(eval_examples, weight_vector, bias, True)
        return None

    def predict_set_in_file(self, test_examples, weight_vector, bias, gen_output):
        correct = 0
        output = []
        file_index = 0
        for ex in test_examples:
            _, true_label, prediction, _ = self.predict_example(ex, weight_vector, bias)
            final_prediction = NEG_LABEL if prediction < 0 else POS_LABEL

            if gen_output:
                flag = final_prediction
                if flag == NEG_LABEL:
                    flag = ZERO_LABEL
                output.append(str(self.ids[file_index]) + "," + str(flag))

            if true_label == final_prediction:
                correct += 1

            file_index += 1

        if gen_output:
            file = open(PREDICTION_PATH + self.output_file_name, "w")
            file.write("Id,Prediction\n")
            for item in output:
                file.write("%s\n" % item)
        return correct, len(test_examples)

    """
    #########################################################################################################
    RUN PERCEPTRON BASE CASE
    - Handles Simple, Dynamic Learning Rate, Margin and Aggressive Margin Perceptons
    - Uses Flags to handle different variants
    ##########################################################################################################
    """

    def run_base_case_perceptron(self, examples, weight_vector, bias, learning_rate, margin, is_prediction_only,
                                 is_dynamic_learning, time_step, initial_learning_rate, is_aggressive, generate_output):
        total, correct, wrong = 0, 0, 0
        file_index = 0
        output_file = []
        for example in examples:
            # Start Perceptron Training
            total += 1
            features, true_label, prediction, sign_calc = self.predict_example(example, weight_vector, bias)
            if not generate_output:
                success = False if sign_calc <= 0 else True
                if success:
                    correct += 1
                else:
                    wrong += 1
            else:
                flag = 0
                if prediction < 0:
                    flag = 0
                else:
                    flag = 1
                output_file.append(str(self.ids[file_index]) + "," + str(flag))

            # Margin Perceptron
            if not is_prediction_only and margin != 0 and sign_calc < margin:
                # Ongoing Training. Not Prediction. Do Weight Update
                weight_vector, bias = self.update_weight_vector(weight_vector, features, bias, true_label,
                                                                learning_rate, is_aggressive, sign_calc, margin)

            # Normal Weight Update
            if not is_prediction_only and margin == 0 and not success:
                # Ongoing Training. Not Prediction. Do Weight Update
                weight_vector, bias = self.update_weight_vector(weight_vector, features, bias, true_label,
                                                                learning_rate, is_aggressive, sign_calc, margin)

            # Update Time stamp at each step
            if is_dynamic_learning:
                # Dynamic Learning Rate Perceptron
                time_step += 1
                learning_rate = float(initial_learning_rate) / float(1 + time_step)

            file_index += 1

        accuracy = (correct / float(total)) * 100
        if generate_output:
            file = open(PREDICTION_PATH + self.output_file_name, "w")
            file.write("Id,Prediction\n")
            for item in output_file:
                file.write("%s\n" % item)
        return learning_rate, accuracy, weight_vector, bias, time_step

    # Method to update Weight Vector
    def update_weight_vector(self, weight_vector, features, bias, true_label, learning_rate, is_aggressive, sign_calc,
                             margin):
        self.weight_update_count += 1
        if is_aggressive:
            learning_rate_to_use = self.get_aggressive_learning_rate(sign_calc, margin, features)
        else:
            learning_rate_to_use = learning_rate
        count = len(features)
        for i in range(count):
            w = weight_vector[i]
            x = self.transform_log(features[i])
            weight_vector[i] = w + (learning_rate_to_use * true_label * x)

        bias = bias + (learning_rate_to_use * true_label)
        return weight_vector, bias

    # Method to calculate Aggressive Margin Learning Rate
    def get_aggressive_learning_rate(self, sign, margin, features):
        x = n.array(features)
        xt = n.array(features)
        numerator = margin - sign
        denomenator = x.dot(xt) + 1
        return float(numerator) / float(denomenator)

    # Method to run Cross-validation and determine best learning rate
    def get_best_learning_rate(self, learning_rates, is_dynamic_learning_rate, margins, is_aggressive):
        CROSS_VALIDATION_EPOCHS = 10
        time_step = DEFAULT_TIME_STEP
        accuracy_list = []
        for margin in margins:
            for learning_rate in learning_rates:
                initial_learning_rate = learning_rate
                total_accuracy = 0.0
                log(
                    "-----------------------------------------------------------------------------------------------------------------------")
                if margin != 0: log("Testing for Margin : " + str(margin))
                if not is_aggressive:  log("Testing for Learning Rate : " + str(learning_rate))
                all_examples = self.get_file_entries(TRAIN_F)
                breakpoint = len(all_examples) / 5
                cross_exs = [all_examples[:breakpoint], all_examples[breakpoint: 2 * breakpoint],
                             all_examples[2 * breakpoint: 3 * breakpoint], all_examples[3 * breakpoint: 4 * breakpoint],
                             all_examples[4 * breakpoint:]]
                for i in range(5):
                    test_entries = cross_exs[i]
                    train_entries = [exs for exs in cross_exs if exs != test_entries]
                    init_array = [rand() for i in range(SIZE_OF_FEATURES)]
                    weight_vector = n.array(init_array)
                    bias = rand()
                    for j in range(CROSS_VALIDATION_EPOCHS):
                        for exs in train_entries:
                            learning_rate, accuracy, weight_vector, bias, ts = self.run_base_case_perceptron(exs,
                                                                                                             weight_vector,
                                                                                                             bias,
                                                                                                             learning_rate,
                                                                                                             margin,
                                                                                                             False,
                                                                                                             is_dynamic_learning_rate,
                                                                                                             time_step,
                                                                                                             initial_learning_rate,
                                                                                                             is_aggressive,
                                                                                                             None)
                            if is_dynamic_learning_rate:
                                time_step = ts

                    # Cross-validation prediction
                    correct, total = self.predict_set_in_file(test_entries, weight_vector, bias, False)
                    accuracy = correct / float(total)
                    total_accuracy += accuracy
                average_accuracy = total_accuracy / float(5)
                accuracy_list.append((margin, initial_learning_rate, average_accuracy))
                log("MARGIN : " + str(margin) + " RATE : " + str(initial_learning_rate) + " ACCURACY : " + str(
                    average_accuracy))

        log(
            "-----------------------------------------------------------------------------------------------------------------------")
        return max(accuracy_list, key=lambda item: item[2])

    def set_output_file_name(self, output_file_name):
        self.output_file_name = output_file_name;
        return

    def get_file_entries(self, file_name):
        file = open(file_name, "r")
        entries = []
        for line in file:
            line = line.rstrip().rstrip("\n")
            entries.append(line)
        return entries

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

    def predict_example(self, example, weight_vector, bias):
        tokens = example.split()
        true_label = int(tokens[0])
        if true_label == 0:
            true_label = NEG_LABEL
        features = [0.0] * SIZE_OF_FEATURES
        modified_tokens = [(token.split(":")[0], token.split(":")[1]) for token in tokens if ":" in token]
        for token in modified_tokens:
            index, value = int(token[0]), float(token[1])
            features[index - 1] = self.transform_log(value)

        features = n.array(features)
        prediction = n.array(weight_vector).dot(features) + bias
        sign_calc = true_label * prediction
        return features, true_label, prediction, sign_calc

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



if __name__ == '__main__':
    print("Run from Main2.py")
