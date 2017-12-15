import regression as Reg
print("################################")
print("##### Logistic Regression")
print("################################")
#### Cross-Validation to determine best hyper parameters
print "Finding Best Hyper parameters by Cross- Validation"
print("-------------------------------------------------------------------------------")
LEARNING_RATES = [10, 1, 0.1, 0.01, 0.001, 0.0001]
LOSS_PARAMS = [10, 1, 0.1, 0.01, 0.001, 0.0001]
TRAIN_F = "../../data/data-splits/data.train"
TEST_F = "../../data/data-splits/data.test"
reg = Reg.Regression()
best = reg.get_best_learning_rate(LEARNING_RATES, LOSS_PARAMS, TRAIN_F)
# gamma = 0.001
# const = 10
gamma, const = best[1], best[0]
print "Best Hyper Parameters - GAMMA : " + str(gamma) + " BEST CONSTANT : " + str(const)
train_exs = reg.get_file_entries(TRAIN_F)
test_exs = reg.get_file_entries(TEST_F)
reg.set_output_file_name("reg_gamma_" + str(gamma) + "_const_" + str(const) + ".csv")
test_acc = reg.enter_reg(train_exs, test_exs, gamma, const, True)
print("Test Accuracy : " + str(test_acc))
print("-------------------------------------------------------------------------------")
