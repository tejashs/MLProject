import svm_sci as SVM

def run_this():
    print("################################")
    print("##### SVM")
    print("################################")
    print("-------------------------------------------------------------------------------")
    NUM_OF_PREDICTORS = 1
    TRAIN_F = "../../data/data-splits/RandomForest_Transformed.data.train"
    EVAL_F = "../../data/data-splits/RandomForest_Transformed.data.eval"
    svm = SVM.svm()
    gamma = 0.001
    const = 10
    print "GAMMA : " + str(gamma) + " BEST CONSTANT : " + str(const)
    train_exs = svm.get_file_entries(TRAIN_F)
    test_exs = train_exs
    eval_exs = svm.get_file_entries(EVAL_F)
    svm.set_output_file_name("random_forest_svm_bagged_gamma_" + str(gamma) + "_const_" + str(const) + ".csv")
    svm.eval_entries = eval_exs
    svm.enter_svm(train_exs, test_exs, gamma, const, True, NUM_OF_PREDICTORS)
    print("-------------------------------------------------------------------------------")
