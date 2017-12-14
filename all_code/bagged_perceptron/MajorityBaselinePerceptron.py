class MajorityBaselinePerceptron:
    def __init__(self):
        self.bias = 0.0
        self.f_train = "phishing.train"
        self.f_test = "phishing.test"
        self.f_dev = "phishing.dev"

    def start_majority_classifier(self):
        print"-----------------------------------------------------------------------"
        pos, neg = 0, 0
        majority = None
        file = open(self.f_train, "r")
        for line in file:
            # Start Perceptron Training
            tokens = line.split()
            true_label = int(tokens[0])
            if true_label == 1 :
                pos+=1
            else :
                neg+=1

        majority = 1 if pos > neg else -1
        self.predict(majority, self.f_dev, True)
        self.predict(majority, self.f_test, False)


    def predict(self, majority, file, is_dev):
        file = open(file, "r")
        correct, wrong = 0, 0
        for line in file:
            # Start Perceptron Training
            tokens = line.split()
            true_label = int(tokens[0])
            if true_label == majority:
                correct += 1
            else:
                wrong += 1

        accuracy = (float(correct) / (correct+wrong)) * 100
        if is_dev:
            print("Accuracy for Development Set : " + str(accuracy))
        else :
            print("Accuracy for Test Set : " + str(accuracy))

        print"-----------------------------------------------------------------------"