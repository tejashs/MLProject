import decision_tree as DT
import process as P
import datetime

# new_train, new_test, averages = P.Pre_Process().process_file("../data/data-splits/data.train", None, "../data/data-splits/data.test")
print("------------------------------------------------------")
print("Start time")
print("------------------------------------------------------")
print(datetime.datetime.now())
print("------------------------------------------------------")
print("Running Decision Tree for Depth 3, for 1000 trees")
print("------------------------------------------------------")
dt = DT.Decision_Tree()
# dt.start(10, new_train, new_test)
dt.start(1, "../data/data-splits/data.train1.transformed2", "../data/data-splits/data.test1.transformed2")
print("------------------------------------------------------")

print("End time")
print("------------------------------------------------------")
print(datetime.datetime.now())
print("------------------------------------------------------")
