import pandas as pd
class Pre_Process:
    def process_file(self, f_train, f_test):
        # Remove Colons for features
        new_train = self.preprocess(f_train)
        new_test = self.preprocess(f_test)
        self.get_averages(new_train)

    def preprocess(self, file_name):
        read_file = open(file_name, "r")
        f = file_name+"1"
        for line in read_file:
            index = 0
            tokens = line.split(" ")
            str_to_write = ""
            for token in tokens:
                if index == 0:
                    str_to_write+=token
                    index+=1
                    continue
                str_to_write+=" "
                features = token.split(":")
                str_to_write+=features[1]
            str_to_write+="\n"

            str_to_write = str_to_write.lstrip(" ")
            write_file = open(f, "w")
            write_file.write("%s" % str_to_write)
            return f


    def get_averages(self, file_name):
        df = pd.read_csv(file_name, sep='\s', header=None, engine='python')
        to_print = ""
        for i in range(17):
            avg = df[i].mean()
            max = df[i].max()
            min = df[i].min()
            to_print += "Avg - " + str(avg) + ", " + "Max - " + str(max) + ", " + "Min - " + str(min) + ", "
            print(to_print)

if __name__ == '__main__':
    Pre_Process().process_file("../data/data-splits/data.train", "../data/data-splits/data.test")
