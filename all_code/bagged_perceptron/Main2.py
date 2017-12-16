import Perceptron as P
import sys
DEFAULT_MARGIN = 0
DUMMY_LEARNING_RATE = 1
LEARNING_RATES = [1, 0.1, 0.01]
MARGINS = [1, 0.1, 0.01, 0.5]
perceptron = P.Perceptron()
print("################################")
print("##### DYNAMIC LEARNING RATE PERCEPTRON WITH BAGGING")
print("################################")
### Cross-Validation to determine best hyper parameter
print "Finding Best Hyper parameter by Cross- Validation"
IS_DYNAMIC_LEARNING = True
IS_AGGRESSIVE = True
dyn_best_rate = perceptron.get_best_learning_rate(LEARNING_RATES, IS_DYNAMIC_LEARNING, [DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## DYNAMIC LEARNING RATE PERCEPTRON WITH BAGGING")
print "Best Hyper Parameters - " +" LEARNING RATE : "+ str(dyn_best_rate[1]) +" BEST CROSS VALIDATION ACCURACY : " + str(dyn_best_rate[2])
#####-------------------------------------------------------------------------------------------------------------------
print("-------------------------------------------------------------------------------")
file_name="Perceptron_BAGGING_DYNAMIC_lr_"+str(dyn_best_rate[1])+".csv"
perceptron.set_output_file_name(file_name)
perceptron.start_perceptron2(dyn_best_rate[1], IS_DYNAMIC_LEARNING,[DEFAULT_MARGIN], IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")


print("################################")
print("##### AGGRESSIVE MARGIN PERCEPTRON WITH BAGGING")
print("################################")
#### Cross-Validation to determine best hyper parameters
print "Finding Best Hyper parameters by Cross- Validation"
print "Will calculate Aggressive Learning Rate for Weight Updates"
agg_marg_best_rate = perceptron.get_best_learning_rate([DUMMY_LEARNING_RATE], IS_DYNAMIC_LEARNING, MARGINS, IS_AGGRESSIVE)
print("-------------------------------------------------------------------------------")
print("########## AGGRESSIVE MARGIN PERCEPTRON WITH BAGGING")
print "Best Hyper Parameters - MARGIN : " + str(agg_marg_best_rate[0]) + " BEST CROSS VALIDATION ACCURACY : " + str(agg_marg_best_rate[2])
print("-------------------------------------------------------------------------------")
IS_AGGRESSIVE = True
IS_DYNAMIC_LEARNING = True
### Running Training and Prediction using the BEST HYPER PARAMETER - Margin - 0.1
file_name="Perceptron_BAGGING_AGG_margin_"+str(agg_marg_best_rate[0])+".csv"
perceptron.set_output_file_name(file_name)
perceptron.start_perceptron2(DUMMY_LEARNING_RATE, IS_DYNAMIC_LEARNING,[agg_marg_best_rate[0]], IS_AGGRESSIVE)
