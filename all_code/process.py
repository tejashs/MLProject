import pandas as pd
import math as M

POS_LABEL = 1
NEG_LABEL = 0


class Pre_Process:
    def process_file(self, f_train, f_dev, f_test):
        # Remove Colons for features
        new_train = self.preprocess(f_train) if f_train is not None else None
        new_dev = self.preprocess(f_dev) if f_dev is not None else None
        new_test = self.preprocess(f_test) if f_test is not None else None
        averages = self.get_averages(new_train) if new_train is not None else None
        # print(len(averages))
        train_trans = self.transform_file(new_train, averages)
        test_trans = self.transform_file(new_test, averages)
        return train_trans, test_trans, averages

    def preprocess(self, file_name):
        read_file = open(file_name, "r")
        f = file_name + "1"
        write_file = open(f, "w")
        for line in read_file:
            index = 0
            tokens = line.split(" ")
            str_to_write = ""
            for token in tokens:
                if index == 0:
                    str_to_write += token
                    index += 1
                    continue
                str_to_write += " "
                features = token.split(":")
                str_to_write += features[1]
                str_to_write = str_to_write.lstrip(" ")
            write_file.write("%s" % str_to_write)
        return f

    def get_averages(self, file_name):
        df = pd.read_csv(file_name, sep='\s', header=None, engine='python')
        features = []
        for i in range(17):
            if i == 0: continue
            to_print = ""
            avg = df[i].mean()
            max = df[i].max()
            min = df[i].min()
            median = df[i].median()
            mode = df[i].mode()
            offset = M.fabs(min)+1 if min < 0 else 0
            scale = 100 if max < 20 else 1
            to_print += "Avg - " + str(avg) + ", " + "Max - " + str(max) + ", " + "Min - " + str(min) + ", " + "Median - " + str(median) + ", " + "Mode - " + str(mode) + ", "
            print(to_print)
            features.append((min, max, median, avg))
        return features

    def transform_file(self, file_name, values):
        read_file = open(file_name, "r")
        f = file_name + ".transformed2"
        write_file = open(f, "w")
        for line in read_file:
            to_write = ""
            tokens = line.split(" ")
            index = 0
            for token in tokens:
                if index == 0:
                    to_write += token
                    index += 1
                    continue
                to_write += " "
                value = values[index-1]
                to_write += str(index)
                to_write += ":"
                to_write += str(self.get_transformed_val(value, token))

                index += 1
            write_file.write("%s \n" % to_write)
        return f



    def get_transformed_val(self, values, token):
        min = float(values[0])
        max = float(values[1])
        median = float(values[2])
        avg = float(values[3])
        val = float(token)
        if val < median:
            return NEG_LABEL
        else:
            return POS_LABEL





if __name__ == '__main__':
    Pre_Process().process_file("../data/data-splits/data.train", None, "../data/data-splits/data.test")

