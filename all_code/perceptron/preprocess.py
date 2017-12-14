import pandas as pd
from sklearn.datasets import load_svmlight_file

# df = pd.read_csv("../data/data-splits/data.train.csv", sep='\s', header=None, engine='python')
# matched_index = df[df.loc[:, 0] == 0].index
# df.loc[matched_index, 0] = -1
# df.to_csv("../data/data-splits/data2.train.csv", sep=' ', header=None, index=False)
#
# df = pd.read_csv("../data/data-splits/data.test", sep='\s', header=None, engine='python')
# matched_index = df[df.loc[:, 0] == 0].index
# df.loc[matched_index, 0] = -1
# df.to_csv("../data/data-splits/data2.test", sep=' ', header=None, index=False)

for i in range(5):
    file_name = "../data/data-splits/training0" + str(i)
    df = pd.read_csv(file_name+".data", sep='\s', header=None, engine='python')
    matched_index = df[df.loc[:, 0] == 0].index
    df.loc[matched_index, 0] = -1
    df.to_csv(file_name+".csv", sep=' ', header=None, index=False)



# read_file = open("../data/data-splits/data.test", "r")
# write_file = open("../data/data-splits/blah.test.csv", "w")
# for line in read_file:
#     index = 0
#     tokens = line.split(" ")
#     str_to_write = ""
#     for token in tokens:
#         if index == 0:
#             str_to_write+=token
#             index+=1
#             continue
#         str_to_write+=" "
#         features = token.split(":")
#         str_to_write+=features[1]
#
#     str_to_write = str_to_write.lstrip(" ")
#     write_file.write("%s" % str_to_write)


# df = pd.read_csv("../data/data-splits/blah.train.csv", sep='\s', header=None, engine='python')
# to_print = ""
# for i in range(17):
#     avg = df[i].mean()
#     print str(i) + " -- " + str(avg)
#     to_print += str(avg)+", "
#
# print to_print
