import id3 as DT
import start_svm as MainSVM
import datetime

print("------------------------------------------------------")
print("RANDOM FOREST INTO SVM")
print("------------------------------------------------------")
print("Part 1 : Random Forest - Decision Trees With Bagging")
print("------------------------------------------------------")
print("Start time - " + str(datetime.datetime.now()))
print("------------------------------------------------------")
dt = DT.main()
print("------------------------------------------------------")
print("Part 2 : SVM")
print("------------------------------------------------------")
MainSVM.run_this()
print("------------------------------------------------------")
print("End time - " + str(datetime.datetime.now()))
print("------------------------------------------------------")
