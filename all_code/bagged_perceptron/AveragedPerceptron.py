import numpy as n
import random

SIZE_OF_FEATURES = 69
NUM_OF_EPOCHS = 20
RANDOM_SEED = 10
SHOULD_PRINT = True
CROSS_VALIDATION_FILES = ["../data/data-splits/training00.data", "../data/data-splits/training01.data", "../data/data-splits/training02.data", "../data/data-splits/training03.data", "../data/data-splits/training04.data"]
PREDICTION_PATH="../data/data-splits/"

def log(string):
    if SHOULD_PRINT:
        print(string)

def rand():
    return random.uniform(-0.01, 0.01)


class AveragedPerceptron:
    def __init__(self):
        self.f_train = "../data/data-splits/data2.train.csv"
        self.f_test = "../data/data-splits/data2.test"
        self.f_dev = "../data/data-splits/data2.test"
        self.id_file= "../data/data-splits/data.eval.id"
        self.eval_file= "../data/data-splits/data.eval.anon"
        f = open(self.id_file, "r")
        self.ids = [x.strip('\n') for x in f]
        self.weight_update_count = 0
        # random.seed(RANDOM_SEED)

    def start_perceptron(self, learning_rate):
        init_array = [rand() for i in range(SIZE_OF_FEATURES)]
        weight_vector = n.array(init_array)
        avg_weight_vector = n.array(weight_vector[:])
        bias = rand()
        avg_bias = bias
        epoch_weight_dict = []
        self.weight_update_count = 0
        # Run Training
        log("Running Training")
        log("Initial Learning Rate : " + str(learning_rate))
        for epoch in range(NUM_OF_EPOCHS):
            learning_rate, accuracy, weight_vector, bias, avg_weight_vector, avg_bias = self.run_base_case_perceptron(self.f_train, weight_vector, bias, avg_weight_vector, avg_bias,
                                                                                             learning_rate, False, None)
            # Predict on Dev Set
            lr, dev_accuracy, wv, b, avg_weight_vector, avg_bias = self.run_base_case_perceptron(
                self.f_train, weight_vector, bias, avg_weight_vector, avg_bias,
                learning_rate, True, None)

            epoch_weight_dict.append((lr, dev_accuracy, avg_weight_vector, avg_bias, epoch))
            log("--------------")
            log("Epoch Number : " + str(epoch+1))
            log("Accuracy on Training Set : " + str(accuracy))
            log("Accuracy on Dev Set : " + str(dev_accuracy))

        max_val = max(epoch_weight_dict, key=lambda item: item[1])
        best_learning_rate, max_accuracy, best_weight_vector, best_bias, best_epoch = max_val
        log("----------------------")
        log("***********************")
        log("Best Dev Accuracy : " + str(max_accuracy) + " Epoch : " + str(best_epoch+1))
        # Predict on Test Set
        lr, test_set_accuracy, wv, b, avg_wv, avg_b = self.run_base_case_perceptron(self.eval_file, None, None, best_weight_vector, best_bias,
                                                                         best_learning_rate, True, True)
        log("Test Set Accuracy : " + str(test_set_accuracy))
        log("***********************")
        return max_accuracy, test_set_accuracy, self.weight_update_count

    """
    #########################################################################################################
    RUN PERCEPTRON BASE CASE
    ##########################################################################################################
    """

    def run_base_case_perceptron(self, file_name, weight_vector, bias, avg_weight_vector, avg_bias, learning_rate, is_prediction_only, generate_output):
        total, correct, wrong = 0, 0, 0
        file = open(file_name, "r")
        file_index=0;
        output_file= []
        for line in file:
            # Start Perceptron Training
            total += 1
            tokens = line.split()
            true_label = int(tokens[0])
            features = [0.0] * SIZE_OF_FEATURES
            modified_tokens = [(token.split(":")[0], token.split(":")[1]) for token in tokens if ":" in token]
            for token in modified_tokens:
                index, value = int(token[0]), float(token[1])
                features[index] = value

            features = n.array(features)
            prediction = n.array(avg_weight_vector).dot(features) + avg_bias

            if true_label * prediction <= 0:
                wrong +=1
                if not is_prediction_only:
                    # Ongoing Training. Not Prediction. Do Weight Update
                    weight_vector, bias = self.update_weight_vector(weight_vector, features, bias, true_label, learning_rate)
            else :
                correct += 1

            if not is_prediction_only:
                # Ongoing Training. Not Prediction. Do Weight Update
                avg_weight_vector = avg_weight_vector + weight_vector
                avg_bias = avg_bias + bias

            if generate_output:
                flag = 0
                if prediction < 0:
                    flag = 0
                else:
                    flag = 1
                output_file.append(str(self.ids[file_index]) + "," + str(flag))

            file_index+=1;

        accuracy = (correct / float(total)) * 100
        if generate_output:
            file = open(PREDICTION_PATH + self.output_file_name, "w")
            file.write("Id,Prediction\n")
            for item in output_file:
                file.write("%s\n" % item)
        return learning_rate, accuracy, weight_vector, bias, avg_weight_vector, avg_bias

    # SIMPLE PERCEPTRON
    # Method to update Weight Vector
    def update_weight_vector(self, weight_vector, features, bias, true_label, learning_rate):
        self.weight_update_count +=1
        count = len(features)
        for i in range(count):
            w = weight_vector[i]
            x = features[i]
            weight_vector[i] = w + (learning_rate * true_label * x)

        bias = bias + (learning_rate * true_label)
        return weight_vector, bias


# SIMPLE PERCEPTRON
# Method to run Cross-validation and determine best learning rate

    def get_best_learning_rate(self, learning_rates):
        CROSS_VALIDATION_EPOCHS = 10
        accuracy_list= []
        for rate in learning_rates:
            total_accuracy = 0.0
            log("-----------------------------")
            log("Testing for Learning Rate : " + str(rate))
            for i in range(5):
                test_file = CROSS_VALIDATION_FILES[i]
                train_files = [f for f in CROSS_VALIDATION_FILES if f != test_file]
                init_array = [rand() for i in range(SIZE_OF_FEATURES)]
                weight_vector = n.array(init_array)
                avg_weight_vector = n.array(weight_vector[:])
                bias = rand()
                avg_bias = bias
                for f in train_files:
                    for epoch in range(CROSS_VALIDATION_EPOCHS):
                        lr, acc, weight_vector, bias, avg_weight_vector, avg_bias = self.run_base_case_perceptron(f, weight_vector, bias, avg_weight_vector, avg_bias, rate, False, None)


                # Cross-validation prediction
                lr, accuracy, weight_vector, bias, avg_weight_vector, avg_bias = self.run_base_case_perceptron(test_file, weight_vector, bias, avg_weight_vector, avg_bias, rate, True, None)
                log("Running Cross Validation : Test File - " + str(test_file) + " Training Files - " + str(train_files) + "---- Accuracy : " + str(accuracy))
                total_accuracy += accuracy
            average_accuracy = total_accuracy/float(5)
            accuracy_list.append((rate, average_accuracy))
            log(" RATE : "+ str(rate) +" ACCURACY : " + str(average_accuracy))

        log("-----------------------------")
        return max(accuracy_list, key=lambda item:item[1])

    def set_output_file_name(self, output_file_name):
        self.output_file_name = output_file_name;
        return
