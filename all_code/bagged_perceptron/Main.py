import Perceptron as P
import AveragedPerceptron as AP
import MajorityBaselinePerceptron as MBP

DEFAULT_MARGIN = 0
DUMMY_LEARNING_RATE = 1
LEARNING_RATES = [1, 0.1, 0.01]
MARGINS = [1, 0.1, 0.01]
perceptron = P.Perceptron()

#######-----------------------------------------------------------------------------------------------------------------
print("################################")
print("##### MAJORITY BASELINE PERCEPTRON")
print("################################")
MBP.MajorityBaselinePerceptron().start_majority_classifier()
print("-------------------------------------------------------------------------------")


IS_AGGRESSIVE = False
IS_DYNAMIC_LEARNING = False
print("################################")
print("##### SIMPLE PERCEPTRON")
print("################################")
### Cross-Validation to determine best hyper parameter
print "Finding Best Hyper parameter by Cross- Validation"
simple_best_rate = perceptron.get_best_learning_rate(LEARNING_RATES, IS_DYNAMIC_LEARNING, [DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## SIMPLE PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(simple_best_rate[1]) +" BEST CROSS VALIDATION ACCURACY : " + str(simple_best_rate[2])
print("-------------------------------------------------------------------------------")
######------------------------------------------------------------------------------------------------------------------
#### Running Training and Prediction using the BEST HYPER PARAMETER - 0.01
dev_set_acc, test_set_acc, weight_update_count = perceptron.start_perceptron(simple_best_rate[1], IS_DYNAMIC_LEARNING,[DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("#### FINAL RESULTS - SIMPLE PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(simple_best_rate[1])
print "Best Cross Validation Accuracy : " + str(simple_best_rate[2])
print "Total number of updates on Training Set : " + str(weight_update_count)
print "Development Set Accuracy : " + str(dev_set_acc)
print "Test Set Accuracy : " + str(test_set_acc)
print("-------------------------------------------------------------------------------")




print("################################")
print("##### DYNAMIC LEARNING RATE PERCEPTRON")
print("################################")
### Cross-Validation to determine best hyper parameter
print "Finding Best Hyper parameter by Cross- Validation"
IS_DYNAMIC_LEARNING = True
dyn_best_rate = perceptron.get_best_learning_rate(LEARNING_RATES, IS_DYNAMIC_LEARNING, [DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## DYNAMIC LEARNING RATE PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(dyn_best_rate[1]) +" BEST CROSS VALIDATION ACCURACY : " + str(dyn_best_rate[2])
#####-------------------------------------------------------------------------------------------------------------------
print("-------------------------------------------------------------------------------")
IS_DYNAMIC_LEARNING = True
### Running Training and Prediction using the BEST HYPER PARAMETER - 1
dev_set_acc, test_set_acc, weight_update_count = perceptron.start_perceptron(dyn_best_rate[1], IS_DYNAMIC_LEARNING,[DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("#### FINAL RESULTS - DYNAMIC LEARNING RATE PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(dyn_best_rate[1])
print "Best Cross Validation Accuracy : " + str(dyn_best_rate[2])
print "Total number of updates on Training Set : " + str(weight_update_count)
print "Development Set Accuracy : " + str(dev_set_acc)
print "Test Set Accuracy : " + str(test_set_acc)
print("-------------------------------------------------------------------------------")




print("################################")
print("##### MARGIN PERCEPTRON")
print("################################")
#### Cross-Validation to determine best hyper parameters
print "Finding Best Hyper parameters by Cross- Validation"
margin_best_rate = perceptron.get_best_learning_rate(LEARNING_RATES, IS_DYNAMIC_LEARNING, MARGINS, IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## MARGIN PERCEPTRON")
print "Best Hyper Parameters - MARGIN : " + str(margin_best_rate[0]) +" LEARNING RATE : "+ str(margin_best_rate[1]) +" BEST CROSS VALIDATION ACCURACY : " + str(margin_best_rate[2])
#####-------------------------------------------------------------------------------------------------------------------
print("-------------------------------------------------------------------------------")
### Running Training and Prediction using the BEST HYPER PARAMETER - Margin - 0.01 / Rate - 0.1
dev_set_acc, test_set_acc, weight_update_count = perceptron.start_perceptron(margin_best_rate[1], IS_DYNAMIC_LEARNING, [margin_best_rate[0]], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("#### FINAL RESULTS - MARGIN PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(margin_best_rate[1]) + " MARGIN : " + str(margin_best_rate[0])
print "Best Cross Validation Accuracy : " + str(margin_best_rate[2])
print "Total number of updates on Training Set : " + str(weight_update_count)
print "Development Set Accuracy : " + str(dev_set_acc)
print "Test Set Accuracy : " + str(test_set_acc)
print("-------------------------------------------------------------------------------")


print("################################")
print("##### AVERAGED PERCEPTRON")
print("################################")
### Cross-Validation to determine best hyper parameters
print "Averaged Perceptron"
print "Finding Best Hyper parameters by Cross- Validation"
avg_best_rate = AP.AveragedPerceptron().get_best_learning_rate(LEARNING_RATES)
print("-------------------------------------------------------------------------------")
print("########## AVERAGED PERCEPTRON")
print "Best Hyper Parameters - LEARNING RATE : "+ str(avg_best_rate[0]) +" BEST CROSS VALIDATION ACCURACY : " + str(avg_best_rate[1])
print("-------------------------------------------------------------------------------")
### Running Training and Prediction using the BEST HYPER PARAMETER - Rate - 0.01
dev_set_acc, test_set_acc, weight_update_count = AP.AveragedPerceptron().start_perceptron(avg_best_rate[0])
print("-------------------------------------------------------------------------------")
print("#### FINAL RESULTS - AVERAGED PERCEPTRON")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(avg_best_rate[0])
print "Best Cross Validation Accuracy : " + str(avg_best_rate[1])
print "Total number of updates on Training Set : " + str(weight_update_count)
print "Development Set Accuracy : " + str(dev_set_acc)
print "Test Set Accuracy : " + str(test_set_acc)
print("-------------------------------------------------------------------------------")




print("################################")
print("##### AGGRESSIVE MARGIN PERCEPTRON")
print("################################")
#### Cross-Validation to determine best hyper parameters
print "Finding Best Hyper parameters by Cross- Validation"
print "Will calculate Aggressive Learning Rate for Weight Updates"
agg_marg_best_rate = perceptron.get_best_learning_rate([DUMMY_LEARNING_RATE], IS_DYNAMIC_LEARNING, MARGINS, IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## AGGRESSIVE MARGIN PERCEPTRON")
print "Best Hyper Parameters - MARGIN : " + str(agg_marg_best_rate[0]) + " BEST CROSS VALIDATION ACCURACY : " + str(agg_marg_best_rate[2])
print("-------------------------------------------------------------------------------")
IS_AGGRESSIVE = True
### Running Training and Prediction using the BEST HYPER PARAMETER - Margin - 0.1
dev_set_acc, test_set_acc, weight_update_count = perceptron.start_perceptron(DUMMY_LEARNING_RATE, IS_DYNAMIC_LEARNING,[agg_marg_best_rate[0]], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("#### FINAL RESULTS - AGGRESSIVE MARGIN PERCEPTRON")
print "Best Hyper Parameters - " + " MARGIN : " + str(agg_marg_best_rate[0])
print "Best Cross Validation Accuracy : " + str(agg_marg_best_rate[2])
print "Total number of updates on Training Set : " + str(weight_update_count)
print "Development Set Accuracy : " + str(dev_set_acc)
print "Test Set Accuracy : " + str(test_set_acc)
print("-------------------------------------------------------------------------------")