import decision_tree as DT
import datetime

# new_train, new_test, averages = P.Pre_Process().process_file("../data/data-splits/data.train", None, "../data/data-splits/data.test")
print("------------------------------------------------------")
print("Start time")
print("------------------------------------------------------")
print(datetime.datetime.now())
dt = DT.Decision_Tree()
# dt.start(10, new_train, new_test)
dt.start(1, "../../data/data-splits/data.train1.transformed2", "../../data/data-splits/data.train1.transformed2")
print("------------------------------------------------------")

print("End time")
print("------------------------------------------------------")
print(datetime.datetime.now())
print("------------------------------------------------------")
